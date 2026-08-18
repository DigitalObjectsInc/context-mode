/**
 * Always-async bash for the Pi coding agent.
 *
 * Re-registers the "bash" tool so every call returns a job handle
 * immediately (the turn ends, the user can keep chatting) and pushes
 * the result back to the agent via `pi.sendUserMessage` when the
 * subprocess exits — waking the agent if idle, or landing at the next
 * turn boundary if mid-turn.
 *
 * Architecture (informed by Sol / gpt-5.6-sol consultation):
 *   - Supervised process, NOT abandoned. The PID + process group are
 *     tracked; session_shutdown and explicit cancel kill the whole tree.
 *   - Six explicit job states: started / running / succeeded / failed /
 *     cancelled / lost. Dependent work must wait for `succeeded`; the
 *     push message states the outcome unambiguously so the model cannot
 *     treat "started" as "passed".
 *   - File-backed output. stdout+stderr stream to ~/.pi/async-bash/<id>.log.
 *     The push message carries a tail; full output is retrieved by reading
 *     the log file. Nothing is held in memory unbounded.
 *   - No timer-polling. Completion is a push event, not a poll.
 *   - No sync path. Every bash call backgrounds; the contract is uniform.
 *
 * The built-in bash tool's schema (command, timeout) is preserved so the
 * model's existing tool-call priors and every prompt/skill that says
 * "use bash" keep working without surgery.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import {
  mkdirSync,
  openSync,
  closeSync,
  existsSync,
  rmSync,
  readFileSync,
  statSync,
} from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Types ────────────────────────────────────────────────────────────

export type JobState =
  | "started"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "lost";

export interface Job {
  id: string;
  command: string;
  cwd: string;
  pid: number;
  state: JobState;
  exitCode: number | null;
  logFile: string;
  startedAt: number;
  endedAt: number | null;
  child: ChildProcess | null;
}

export interface JobHandle {
  jobId: string;
  status: JobState;
  logFile: string;
}

export interface JobStatus {
  jobId: string;
  state: JobState;
  exitCode: number | null;
  command: string;
  startedAt: number;
  endedAt: number | null;
  logFile: string;
  tail: string;
}

// ── Helpers ──────────────────────────────────────────────────────────

const isWindows = process.platform === "win32";

function jobsDir(): string {
  const home = homedir();
  const dir = join(home, ".pi", "async-bash");
  mkdirSync(dir, { recursive: true });
  return dir;
}

function newJobId(): string {
  return "j_" + randomBytes(4).toString("hex");
}

/** Kill an entire process group. On Windows, taskkill /T; on Unix, kill -PGID. */
function killTree(child: ChildProcess): void {
  if (!child.pid) return;
  if (isWindows) {
    try {
      spawn("taskkill", ["/F", "/T", "/PID", String(child.pid)], {
        stdio: "ignore",
        windowsHide: true,
      });
    } catch {
      // best effort
    }
    return;
  }
  try {
    // Negative PID kills the process group (child was spawned detached).
    process.kill(-child.pid, "SIGKILL");
  } catch {
    // Process may have already exited; try direct kill as fallback.
    try {
      child.kill("SIGKILL");
    } catch {
      // best effort
    }
  }
}

/** Read the tail of a file as a string, capped at maxBytes from the end. */
function tailFile(path: string, maxBytes = 8192): string {
  try {
    const stat = statSync(path);
    if (stat.size <= maxBytes) {
      return readFileSync(path, "utf-8");
    }
    // Read the last maxBytes. Use a buffer + fd seek for efficiency on large
    // files, but readFileSync of a slice is simple and correct for our sizes.
    const fd = require("node:fs").openSync(path, "r");
    try {
      const buf = Buffer.alloc(maxBytes);
      require("node:fs").readSync(fd, buf, 0, maxBytes, stat.size - maxBytes);
      return buf.toString("utf-8");
    } finally {
      require("node:fs").closeSync(fd);
    }
  } catch {
    return "";
  }
}

// ── JobSupervisor ────────────────────────────────────────────────────

/**
 * Owns the job registry for the lifetime of the pi process.
 * Spawns commands detached, streams output to a file, fires a callback
 * on completion so the caller can push the result to the agent.
 */
export class JobSupervisor {
  private readonly jobs = new Map<string, Job>();
  /** Fired when a job reaches a terminal state. Caller pushes to agent. */
  private readonly onDone: (job: Job) => void;

  constructor(onDone: (job: Job) => void) {
    this.onDone = onDone;
  }

  /**
   * Start a command as a supervised background job. Returns immediately
   * with a job handle; the caller's turn ends. When the process exits,
   * {@link onDone} fires with the terminal job state.
   */
  start(command: string, cwd: string, env?: NodeJS.ProcessEnv): JobHandle {
    const id = newJobId();
    const logFile = join(jobsDir(), `${id}.log`);

    // Direct-fd stdio + unref: the child writes straight to the log file
    // and does NOT keep the parent's event loop alive. Without this, piped
    // stdio + .on("data") listeners would hold pi's event loop open, so the
    // turn could not end while the job runs.
    const fd = openSync(logFile, "w");
    const child = spawn(command, {
      cwd,
      env: env ?? process.env,
      shell: true,
      stdio: ["ignore", fd, fd],
      detached: !isWindows,
      windowsHide: true,
    });
    closeSync(fd);
    child.unref();

    const job: Job = {
      id,
      command,
      cwd,
      pid: child.pid ?? 0,
      state: child.pid ? "running" : "lost",
      exitCode: null,
      logFile,
      startedAt: Date.now(),
      endedAt: null,
      child,
    };
    this.jobs.set(id, job);

    const settle = (exitCode: number | null, state: JobState) => {
      // Guard on endedAt (not state name) so cancel() — which sets
      // state="cancelled" before killTree fires the exit event — still
      // reaches onDone exactly once. Checking state names would block
      // the legitimate cancelled path from ever firing.
      if (job.endedAt !== null) return; // already terminal — onDone fired
      job.state = state;
      job.exitCode = exitCode;
      job.endedAt = Date.now();
      job.child = null;
      this.onDone(job);
    };

    child.on("exit", (code, signal) => {
      if (signal) {
        // Killed by a signal (SIGTERM from cancel, SIGKILL from shutdown).
        // If we already marked it cancelled, settle already returned.
        settle(null, job.state === "cancelled" ? "cancelled" : "failed");
      } else {
        settle(code, code === 0 ? "succeeded" : "failed");
      }
    });
    child.on("error", (err) => {
      // Spawn itself failed (ENOENT, EACCES, …) — not a command failure.
      try {
        const { appendFileSync } = require("node:fs");
        appendFileSync(logFile, `[spawn error: ${err.message}]\n`);
      } catch {
        // best effort
      }
      settle(null, "failed");
    });

    return { jobId: id, status: job.state, logFile };
  }

  /** Status of a job, including a tail of its output. */
  status(jobId: string): JobStatus | null {
    const job = this.jobs.get(jobId);
    if (!job) return null;
    return {
      jobId: job.id,
      state: job.state,
      exitCode: job.exitCode,
      command: job.command,
      startedAt: job.startedAt,
      endedAt: job.endedAt,
      logFile: job.logFile,
      tail: tailFile(job.logFile),
    };
  }

  /**
   * Read output by byte offset. Returns content + nextOffset + eof.
   * Lets the model pull full output incrementally without dumping it
   * all into context at once.
   */
  read(jobId: string, offset = 0, maxBytes = 32768): { content: string; nextOffset: number; eof: boolean } | null {
    const job = this.jobs.get(jobId);
    if (!job || !existsSync(job.logFile)) return null;
    try {
      const stat = statSync(job.logFile);
      if (offset >= stat.size) {
        return { content: "", nextOffset: offset, eof: true };
      }
      const fd = require("node:fs").openSync(job.logFile, "r");
      try {
        const len = Math.min(maxBytes, stat.size - offset);
        const buf = Buffer.alloc(len);
        require("node:fs").readSync(fd, buf, 0, len, offset);
        const nextOffset = offset + len;
        return {
          content: buf.toString("utf-8"),
          nextOffset,
          eof: nextOffset >= stat.size,
        };
      } finally {
        require("node:fs").closeSync(fd);
      }
    } catch {
      return null;
    }
  }

  /** Cancel a running job: kill the process tree, mark cancelled. */
  cancel(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job || !job.child) return false;
    if (job.state === "succeeded" || job.state === "failed" || job.state === "cancelled") {
      return false; // already terminal
    }
    job.state = "cancelled";
    killTree(job.child);
    return true;
  }

  /** Kill all live jobs. Called on session_shutdown to prevent zombies. */
  cancelAll(): void {
    for (const job of this.jobs.values()) {
      if (job.child && job.state !== "succeeded" && job.state !== "failed" && job.state !== "cancelled") {
        job.state = "cancelled";
        killTree(job.child);
      }
    }
  }

  /** All jobs (for diagnostics / a list tool). */
  list(): JobStatus[] {
    return Array.from(this.jobs.values()).map((j) => this.status(j.id)!).filter(Boolean);
  }

  /** Drop completed jobs older than maxAgeMs to bound memory. */
  prune(maxAgeMs = 24 * 60 * 60 * 1000): void {
    const now = Date.now();
    for (const [id, job] of this.jobs) {
      const ended = job.endedAt ?? job.startedAt;
      if (now - ended > maxAgeMs) {
        // Best-effort log cleanup.
        try {
          if (existsSync(job.logFile)) rmSync(job.logFile, { force: true });
        } catch {
          // best effort
        }
        this.jobs.delete(id);
      }
    }
  }
}

// ── Push message formatting ───────────────────────────────────────────

/**
 * Format the completion message that gets pushed to the agent via
 * `pi.sendUserMessage`. The state word is unambiguous (SUCCEEDED /
 * FAILED / CANCELLED) so the model cannot treat "started" as "passed".
 */
export function formatResultMessage(job: Job): string {
  const stateLabel = job.state.toUpperCase();
  const exitPart = job.exitCode !== null ? ` (exit ${job.exitCode})` : "";
  const lines: string[] = [
    `[job ${job.id} finished: ${stateLabel}${exitPart}]`,
    "",
    `Command: ${job.command}`,
    "",
    "--- output (tail) ---",
    tailFile(job.logFile) || "(no output)",
    "",
    `Full output: ${job.logFile}`,
  ];
  return lines.join("\n");
}

// ── Pi registration ──────────────────────────────────────────────────

/**
 * The shape of the pi ExtensionAPI we touch. Typed structurally (same
 * style as mcp-bridge.ts) so we don't pull @earendil-works/pi-coding-agent
 * as a build dependency.
 */
export interface PiLikeAPI {
  registerTool: (tool: {
    name: string;
    label?: string;
    description: string;
    parameters: unknown;
    execute: (
      toolCallId: string,
      params: Record<string, unknown>,
      signal: AbortSignal | undefined,
      onUpdate?: unknown,
      ctx?: unknown,
    ) => Promise<{ content: Array<{ type: "text"; text: string }>; details: Record<string, unknown> }>;
    renderCall?: unknown;
    renderResult?: unknown;
  }) => void;
  /**
   * Push a user message to the agent. Always triggers a turn. When the
   * agent is streaming, `deliverAs` controls how the message is queued:
   * "followUp" lands it after the current turn. This is the push path —
   * job completion wakes the agent if idle, or arrives at the turn
   * boundary if mid-turn.
   */
  sendUserMessage?: (
    content: string,
    options?: { deliverAs?: "steer" | "followUp"; expandPromptTemplates?: boolean },
  ) => void;
  logger?: {
    debug?: (m: string) => void;
    info?: (m: string) => void;
    warn?: (m: string) => void;
    error?: (m: string) => void;
  };
}

/** Schema matching the built-in bash tool so model priors are preserved. */
const BASH_PARAMETERS = {
  type: "object",
  properties: {
    command: { type: "string", description: "Bash command to execute" },
    timeout: {
      type: "number",
      description: "Timeout in seconds (optional, no default timeout)",
    },
  },
  required: ["command"],
};

const ASYNC_BASH_DESCRIPTION =
  "Execute a bash command asynchronously. Returns immediately with a job handle " +
  "({jobId, status, logFile}); the command runs in the background. When it finishes, " +
  "the result (exit code + output tail) is delivered automatically as a new message — " +
  "do NOT poll. Read the full log file with the Read tool if you need more than the tail. " +
  "The command's working directory is the current project directory.";

/**
 * Re-register the "bash" tool as always-async. Every call returns a job
 * handle immediately; completion is pushed via `pi.sendUserMessage`.
 *
 * Returns the JobSupervisor so the caller can wire cancelAll() into
 * session_shutdown and prune() into a periodic timer.
 */
export function registerAsyncBash(pi: PiLikeAPI, cwd: () => string): JobSupervisor {
  const supervisor = new JobSupervisor((job) => {
    const msg = formatResultMessage(job);
    if (pi.sendUserMessage) {
      // followUp: triggers a turn if idle; queues after the turn if streaming.
      pi.sendUserMessage(msg, { deliverAs: "followUp", expandPromptTemplates: false });
    } else {
      // No push path available — log so a test harness can observe it.
      pi.logger?.warn?.(`[async-bash] job ${job.id} finished (${job.state}) but sendUserMessage unavailable`);
    }
  });

  pi.registerTool({
    name: "bash",
    label: "bash",
    description: ASYNC_BASH_DESCRIPTION,
    parameters: BASH_PARAMETERS,
    async execute(_toolCallId, params, _signal) {
      const command = String(params.command ?? "");
      if (!command.trim()) {
        return {
          content: [{ type: "text", text: "Error: empty command" }],
          details: { error: true },
        };
      }
      const handle = supervisor.start(command, cwd());
      return {
        content: [
          {
            type: "text",
            text: `Job started: ${handle.jobId}\nStatus: ${handle.status}\nLog: ${handle.logFile}\n\nThe result will arrive automatically when the command finishes. Do not poll.`,
          },
        ],
        details: { jobId: handle.jobId, async: true, logFile: handle.logFile },
      };
    },
  });

  return supervisor;
}

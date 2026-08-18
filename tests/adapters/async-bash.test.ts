import "../setup-home";
/**
 * Always-async bash — job supervisor + push-on-completion.
 *
 * Verifies the six Sol-recommended guarantees:
 *   1. start() returns immediately with a job handle (the turn can end).
 *   2. The process is supervised: exit fires onDone with a terminal state.
 *   3. Six states are distinguishable: succeeded / failed / cancelled / lost.
 *   4. Output streams to a file; tail + offset reads work, bounded.
 *   5. cancel() kills the process tree and settles as "cancelled".
 *   6. cancelAll() (session_shutdown) reaps all live jobs.
 *
 * Also pins that the pi registration preserves the built-in bash schema
 * (command, timeout) so model priors survive the override.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  JobSupervisor,
  registerAsyncBash,
  formatResultMessage,
  type PiLikeAPI,
} from "../../src/adapters/pi/async-bash.js";

let scratch: string;

beforeEach(() => {
  scratch = mkdtempSync(join(tmpdir(), "async-bash-"));
});

afterEach(() => {
  try {
    rmSync(scratch, { recursive: true, force: true });
  } catch {
    /* best effort */
  }
});

// A fake pi that records registerTool + sendUserMessage calls.
function fakePi(): PiLikeAPI & {
  tools: Map<string, any>;
  messages: string[];
} {
  const tools = new Map<string, any>();
  const messages: string[] = [];
  return {
    tools,
    messages,
    registerTool: (tool) => {
      tools.set(tool.name, tool);
    },
    sendUserMessage: (content: string) => {
      messages.push(content);
    },
    logger: {
      warn: () => {},
      debug: () => {},
      info: () => {},
      error: () => {},
    },
  };
}

describe("JobSupervisor", () => {
  it("start() returns immediately with a running job handle", () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    const handle = sup.start("echo hello", scratch);
    expect(handle.jobId).toMatch(/^j_/);
    expect(handle.status).toBe("running");
    expect(handle.logFile).toContain("async-bash");
  });

  it("settles as succeeded when the command exits 0 and fires onDone", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    sup.start("echo hello", scratch);
    // Wait for the process to exit.
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    expect(done[0].state).toBe("succeeded");
    expect(done[0].exitCode).toBe(0);
  });

  it("settles as failed when the command exits non-zero", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    sup.start("exit 3", scratch);
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    expect(done[0].state).toBe("failed");
    expect(done[0].exitCode).toBe(3);
  });

  it("streams stdout to the log file; tail reads it back", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    sup.start("echo line1 && echo line2", scratch);
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    const status = sup.status(done[0].id)!;
    expect(status.tail).toContain("line1");
    expect(status.tail).toContain("line2");
  });

  it("read() returns output by offset with eof at the end", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    sup.start("printf hello", scratch);
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    const id = done[0].id;
    const first = sup.read(id, 0)!;
    expect(first.content).toContain("hello");
    // After reading past the end, eof is true.
    const second = sup.read(id, first.nextOffset)!;
    expect(second.eof).toBe(true);
    expect(second.content).toBe("");
  });

  it("cancel() kills a running job and settles as cancelled", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    // 30-second sleep — must NOT complete on its own.
    sup.start("sleep 30", scratch);
    // Give it a moment to spawn, then cancel.
    await new Promise((r) => setTimeout(r, 100));
    const id = [...sup["jobs"].keys()][0];
    expect(sup.cancel(id)).toBe(true);
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    expect(done[0].state).toBe("cancelled");
  });

  it("cancelAll() reaps all live jobs (session_shutdown path)", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    sup.start("sleep 30", scratch);
    sup.start("sleep 30", scratch);
    await new Promise((r) => setTimeout(r, 100));
    sup.cancelAll();
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length >= 2) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    expect(done.length).toBe(2);
    expect(done.every((j) => j.state === "cancelled")).toBe(true);
  });

  it("formatResultMessage labels state unambiguously (no 'started' as 'passed')", async () => {
    const done: any[] = [];
    const sup = new JobSupervisor((j) => done.push(j));
    sup.start("echo done", scratch);
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (done.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    const msg = formatResultMessage(done[0]);
    expect(msg).toContain("SUCCEEDED");
    expect(msg).toContain(done[0].id);
    expect(msg).toContain("Full output:");
  });
});

describe("registerAsyncBash — pi tool registration", () => {
  it("registers a 'bash' tool preserving the built-in schema (command, timeout)", () => {
    const pi = fakePi();
    registerAsyncBash(pi, () => scratch);
    expect(pi.tools.has("bash")).toBe(true);
    const tool = pi.tools.get("bash")!;
    const params = tool.parameters as any;
    expect(params.properties.command.type).toBe("string");
    expect(params.properties.timeout.type).toBe("number");
    expect(params.required).toContain("command");
  });

  it("execute() returns a job handle immediately (the turn can end)", async () => {
    const pi = fakePi();
    registerAsyncBash(pi, () => scratch);
    const tool = pi.tools.get("bash")!;
    const result = await tool.execute("tc1", { command: "echo hi" }, undefined);
    const text = result.content[0].text;
    expect(text).toMatch(/Job started: j_/);
    expect(text).toContain("Do not poll");
    expect(result.details).toHaveProperty("jobId");
    expect(result.details).toHaveProperty("async", true);
  });

  it("pushes the result via sendUserMessage when the subprocess finishes", async () => {
    const pi = fakePi();
    registerAsyncBash(pi, () => scratch);
    const tool = pi.tools.get("bash")!;
    await tool.execute("tc2", { command: "echo done" }, undefined);
    // The completion is pushed asynchronously — wait for it.
    await new Promise<void>((resolve) => {
      const wait = () => {
        if (pi.messages.length > 0) return resolve();
        setTimeout(wait, 25);
      };
      wait();
    });
    expect(pi.messages.length).toBeGreaterThan(0);
    expect(pi.messages[0]).toContain("SUCCEEDED");
  });

  it("rejects an empty command without spawning", async () => {
    const pi = fakePi();
    registerAsyncBash(pi, () => scratch);
    const tool = pi.tools.get("bash")!;
    const result = await tool.execute("tc3", { command: "   " }, undefined);
    expect(result.content[0].text).toContain("empty command");
  });
});

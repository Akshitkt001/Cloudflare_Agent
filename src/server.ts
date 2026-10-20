import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { convertToModelMessages, pruneMessages, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

export type Result = { topic: string; score: number; at: string };
export type CoachState = { results: Result[] };

export interface Env {
  AI: any;
}

export class ChatAgent extends AIChatAgent<Env, CoachState> {
  initialState: CoachState = { results: [] };
  maxPersistedMessages = 100;
  chatRecovery = true;
  waitForMcpConnections = true;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/meta/llama-3.1-8b-instruct-fp8"),
      system: `You are a friendly but rigorous interview coach for Python backend and AI/ML roles. Rules:
1. Ask ONE question at a time.
2. After the user answers, give a score out of 10 and a short note on what was good and what was missing. Ensure you pass the score parameter as a strict raw number/integer, never a text string.
3. Then call the recordResult tool immediately using real tool execution schema blocks. Do not type out tool block text formats like {"name": ...} in your response text window under any circumstances.
4. When asked about progress, call getProgress and point out the weak topics.
5. If the user wants a reminder or mock interview later, use the scheduleTask tool.
Keep replies short and encouraging.

${getSchedulePrompt({ date: new Date(new Date().toISOString()) })}`,
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        // Dynamic MCP tools from connected servers
        ...mcpTools,

        recordResult: tool({
          description: "Save the score for an interview topic after grading",
          inputSchema: z.object({
            topic: z.string().describe("short topic, e.g. database indexing"),
            score: z.number().describe("Provide a raw integer score from 0 to 10")
          }),
          execute: async ({ topic, score }) => {
            const agentInstance = this;
            const currentResults = agentInstance.state?.results || [];
            const updatedResults = [
              ...currentResults, 
              { topic, score: Math.round(score), at: new Date().toISOString() }
            ];

            agentInstance.setState({
              results: updatedResults
            });
            
            await agentInstance.save(); 
            return `Saved ${topic}: ${score}/10`;
          }
        }),

        getProgress: tool({
          description: "Get saved scores grouped by topic to find weak areas",
          inputSchema: z.object({}),
          execute: async () => {
            const currentResults = this.state?.results || [];
            const by: Record<string, number[]> = {};
            for (const r of currentResults) {
              if (r && r.topic) (by[r.topic] ??= []).push(r.score);
            }
            return Object.entries(by).map(([topic, s]) => ({
              topic,
              attempts: s.length,
              average: Math.round((s.reduce((a, b) => a + b, 0) / s.length) * 10) / 10
            }));
          }
        }),

        // Restored Math Tool
        calculate: tool({
          description: "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z.enum(["+", "-", "*", "/", "%"]).describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) => Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        // Restored Scheduling Tool
        scheduleTask: tool({
          description: "Schedule an interview reminder to be executed at a later time.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") return "Not a valid schedule input";
            const input = when.type === "scheduled" ? when.date : when.type === "delayed" ? when.delayInSeconds : when.type === "cron" ? when.cron : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, { idempotent: true });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;

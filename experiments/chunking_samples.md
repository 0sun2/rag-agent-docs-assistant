# Chunking samples (Phase 2)

- chunk_size = 1000, chunk_overlap = 150
- semantic 적용 파일 수: 5

## Strategy stats (전체 코퍼스)

| strategy | count | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| fixed | 14669 | 1993 | 922 | 2 | 8000 |
| recursive | 35447 | 907 | 1000 | 2 | 1000 |
| markdown | 39426 | 799 | 998 | 2 | 1000 |
| semantic* | 16 | 2973 | 2053 | 441 | 10006 |

\* semantic 은 샘플 5개 파일에만 적용한 결과입니다.

## File: `oss/langchain/frontend/join-rejoin.mdx`
- length: 12303 chars

### [fixed] — 16 chunks
**chunk 0** (666 chars)

```
Join and rejoin lets you disconnect from a running agent stream without stopping the agent, then reconnect to it later. The agent continues executing server-side while the client is away, and you pick up the stream exactly where you left off. ⏎  ⏎ import { PatternEmbed } from "/snippets/pattern-embed.jsx" ⏎  ⏎ <PatternEmbed pattern="join-rejoin" /> ⏎  ⏎ import RequiresLanggraphServer from '/snippets/oss/requires-langgraph-server.mdx'; ⏎  ⏎ <RequiresLanggraphServer /> ⏎  ⏎ ## Why join & rejoin? ⏎ …
```

**chunk 1** (624 chars)

```
- **Network interruptions**: mobile users moving between cell towers or Wi-Fi networks can seamlessly resume ⏎ - **Page navigation**: users navigating away from a chat page and returning later without losing progress ⏎ - **Mobile backgrounding**: apps suspended by the OS can rejoin the stream when foregrounded ⏎ - **Long-running tasks**: agents performing multi-minute operations (research, code generation, data analysis) where users don't need to keep the page open ⏎ - **Multi-device handoff**:  …
```

**chunk 2** (881 chars)

```
## Core concepts ⏎  ⏎ The join/rejoin pattern involves three key mechanisms: ⏎  ⏎ | Method / Option | Purpose | ⏎ |---|---| ⏎ | `stream.stop()` | Disconnect the client from the stream without stopping the agent | ⏎ | `stream.joinStream(runId)` | Reconnect to an existing stream by its run ID | ⏎ | `onDisconnect: "continue"` | Submit option that tells the server to keep running after client disconnects | ⏎ | `streamResumable: true` | Submit option that enables the stream to be rejoined later | ⏎   …
```

**chunk 3** (868 chars)

```
## Setting up `useStream` ⏎  ⏎ The key setup step is capturing the `run_id` from the `onCreated` callback so you can rejoin later. ⏎  ⏎ :::python ⏎  ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type parameter to `useStream` for type-safe access to state values. In the examples below, replace `typeof myAgent` with your interface name: ⏎  ⏎ ```ts ⏎ import type { BaseMessage } from "@langchain/core/messages"; ⏎  ⏎ interface AgentState { ⏎   messages: BaseMessa …
```

### [recursive] — 16 chunks
**chunk 0** (666 chars)

```
Join and rejoin lets you disconnect from a running agent stream without stopping the agent, then reconnect to it later. The agent continues executing server-side while the client is away, and you pick up the stream exactly where you left off. ⏎  ⏎ import { PatternEmbed } from "/snippets/pattern-embed.jsx" ⏎  ⏎ <PatternEmbed pattern="join-rejoin" /> ⏎  ⏎ import RequiresLanggraphServer from '/snippets/oss/requires-langgraph-server.mdx'; ⏎  ⏎ <RequiresLanggraphServer /> ⏎  ⏎ ## Why join & rejoin? ⏎ …
```

**chunk 1** (624 chars)

```
- **Network interruptions**: mobile users moving between cell towers or Wi-Fi networks can seamlessly resume ⏎ - **Page navigation**: users navigating away from a chat page and returning later without losing progress ⏎ - **Mobile backgrounding**: apps suspended by the OS can rejoin the stream when foregrounded ⏎ - **Long-running tasks**: agents performing multi-minute operations (research, code generation, data analysis) where users don't need to keep the page open ⏎ - **Multi-device handoff**:  …
```

**chunk 2** (881 chars)

```
## Core concepts ⏎  ⏎ The join/rejoin pattern involves three key mechanisms: ⏎  ⏎ | Method / Option | Purpose | ⏎ |---|---| ⏎ | `stream.stop()` | Disconnect the client from the stream without stopping the agent | ⏎ | `stream.joinStream(runId)` | Reconnect to an existing stream by its run ID | ⏎ | `onDisconnect: "continue"` | Submit option that tells the server to keep running after client disconnects | ⏎ | `streamResumable: true` | Submit option that enables the stream to be rejoined later | ⏎   …
```

**chunk 3** (868 chars)

```
## Setting up `useStream` ⏎  ⏎ The key setup step is capturing the `run_id` from the `onCreated` callback so you can rejoin later. ⏎  ⏎ :::python ⏎  ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type parameter to `useStream` for type-safe access to state values. In the examples below, replace `typeof myAgent` with your interface name: ⏎  ⏎ ```ts ⏎ import type { BaseMessage } from "@langchain/core/messages"; ⏎  ⏎ interface AgentState { ⏎   messages: BaseMessa …
```

### [markdown] — 17 chunks
**chunk 0** (459 chars)

```
Join and rejoin lets you disconnect from a running agent stream without stopping the agent, then reconnect to it later. The agent continues executing server-side while the client is away, and you pick up the stream exactly where you left off.   ⏎ import { PatternEmbed } from "/snippets/pattern-embed.jsx"   ⏎ <PatternEmbed pattern="join-rejoin" />   ⏎ import RequiresLanggraphServer from '/snippets/oss/requires-langgraph-server.mdx';   ⏎ <RequiresLanggraphServer />
```

**chunk 1** (763 chars)

```
## Why join & rejoin?   ⏎ Traditional streaming APIs tightly couple the client and server: if the client disconnects, the stream is lost. Join and rejoin breaks this coupling, enabling several important patterns:   ⏎ - **Network interruptions**: mobile users moving between cell towers or Wi-Fi networks can seamlessly resume ⏎ - **Page navigation**: users navigating away from a chat page and returning later without losing progress ⏎ - **Mobile backgrounding**: apps suspended by the OS can rejoin  …
```

**chunk 2** (745 chars)

```
## Core concepts   ⏎ The join/rejoin pattern involves three key mechanisms:   ⏎ | Method / Option | Purpose | ⏎ |---|---| ⏎ | `stream.stop()` | Disconnect the client from the stream without stopping the agent | ⏎ | `stream.joinStream(runId)` | Reconnect to an existing stream by its run ID | ⏎ | `onDisconnect: "continue"` | Submit option that tells the server to keep running after client disconnects | ⏎ | `streamResumable: true` | Submit option that enables the stream to be rejoined later |   ⏎ < …
```

**chunk 3** (875 chars)

```
## Setting up `useStream`   ⏎ The key setup step is capturing the `run_id` from the `onCreated` callback so you can rejoin later.   ⏎ :::python   ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type parameter to `useStream` for type-safe access to state values. In the examples below, replace `typeof myAgent` with your interface name:   ⏎ ```ts ⏎ import type { BaseMessage } from "@langchain/core/messages"; ⏎  ⏎ interface AgentState { ⏎ messages: BaseMessage[];  …
```

### [semantic] — 3 chunks
**chunk 0** (1812 chars)

```
Join and rejoin lets you disconnect from a running agent stream without stopping the agent, then reconnect to it later. The agent continues executing server-side while the client is away, and you pick up the stream exactly where you left off. import { PatternEmbed } from "/snippets/pattern-embed.jsx" ⏎  ⏎ <PatternEmbed pattern="join-rejoin" /> ⏎  ⏎ import RequiresLanggraphServer from '/snippets/oss/requires-langgraph-server.mdx'; ⏎  ⏎ <RequiresLanggraphServer /> ⏎  ⏎ ## Why join & rejoin? Tradit …
```

**chunk 1** (10006 chars)

```
The agent continues processing server-side. To actually cancel the agent's execution, you would use interrupt or cancel mechanisms instead. </Note> ⏎  ⏎ ## Setting up `useStream` ⏎  ⏎ The key setup step is capturing the `run_id` from the `onCreated` callback so you can rejoin later. :::python ⏎  ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type parameter to `useStream` for type-safe access to state values. In the examples below, replace `typeof myAgent` wit …
```

**chunk 2** (478 chars)

```
- **Show clear connection state**: users should always know whether they are receiving live updates or viewing a snapshot. - **Auto-rejoin on visibility change**: use the Page Visibility API to automatically rejoin when the user returns to the tab. - **Set reasonable timeouts**: if a rejoin attempt takes too long, fall back to fetching the thread history instead. - **Clean up completed runs**: remove persisted run IDs when the agent finishes to avoid stale rejoin attempts.
```


## File: `oss/langchain/frontend/time-travel.mdx`
- length: 14278 chars

### [fixed] — 17 chunks
**chunk 0** (972 chars)

```
Every state change in a LangGraph agent creates a **checkpoint**, a complete ⏎ snapshot of the agent's state at that moment. Time travel lets you inspect any ⏎ checkpoint, view the exact state the agent held, and **resume execution from ⏎ that point** to explore alternative paths. It's a debugger, an undo button, and ⏎ an audit log all in one. ⏎  ⏎ import { PatternEmbed } from "/snippets/pattern-embed.jsx" ⏎  ⏎ <PatternEmbed pattern="time-travel" /> ⏎  ⏎ import RequiresLanggraphServer from '/sni …
```

**chunk 1** (989 chars)

```
This creates a linear timeline of every decision the agent made, every tool it ⏎ called, and every response it produced. Your UI can render this timeline and let ⏎ users jump to any point. ⏎  ⏎ ## Setting up useStream ⏎  ⏎ Enable checkpoint history by passing `fetchStateHistory: true` to `useStream`. ⏎ This tells the hook to load the full checkpoint timeline for the current thread. ⏎  ⏎ :::python ⏎  ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type paramete …
```

**chunk 2** (979 chars)

```
```ts ⏎ import type { myAgent } from "./agent"; ⏎ ``` ⏎  ⏎ ::: ⏎  ⏎ <CodeGroup> ⏎ ```tsx React ⏎ import { useStream } from "@langchain/react"; ⏎  ⏎ const AGENT_URL = "http://localhost:2024"; ⏎  ⏎ export function TimeTravelChat() { ⏎   const stream = useStream<typeof myAgent>({ ⏎     apiUrl: AGENT_URL, ⏎     assistantId: "time_travel", ⏎     fetchStateHistory: true, ⏎   }); ⏎  ⏎   const history = stream.history ?? []; ⏎  ⏎   return ( ⏎     <div className="flex h-screen"> ⏎       <ChatPanel messag …
```

**chunk 3** (890 chars)

```
const history = computed(() => stream.history.value ?? []); ⏎  ⏎ function resumeFrom(cp: ThreadState) { ⏎   stream.submit(null, { checkpoint: cp.checkpoint }); ⏎ } ⏎ </script> ⏎  ⏎ <template> ⏎   <div class="flex h-screen"> ⏎     <ChatPanel :messages="stream.messages.value" /> ⏎     <TimelineSidebar :history="history" @select="resumeFrom" /> ⏎   </div> ⏎ </template> ⏎ ``` ⏎  ⏎ ```svelte Svelte ⏎ <script lang="ts"> ⏎   import { useStream } from "@langchain/svelte"; ⏎  ⏎   const AGENT_URL = "http: …
```

### [recursive] — 17 chunks
**chunk 0** (972 chars)

```
Every state change in a LangGraph agent creates a **checkpoint**, a complete ⏎ snapshot of the agent's state at that moment. Time travel lets you inspect any ⏎ checkpoint, view the exact state the agent held, and **resume execution from ⏎ that point** to explore alternative paths. It's a debugger, an undo button, and ⏎ an audit log all in one. ⏎  ⏎ import { PatternEmbed } from "/snippets/pattern-embed.jsx" ⏎  ⏎ <PatternEmbed pattern="time-travel" /> ⏎  ⏎ import RequiresLanggraphServer from '/sni …
```

**chunk 1** (989 chars)

```
This creates a linear timeline of every decision the agent made, every tool it ⏎ called, and every response it produced. Your UI can render this timeline and let ⏎ users jump to any point. ⏎  ⏎ ## Setting up useStream ⏎  ⏎ Enable checkpoint history by passing `fetchStateHistory: true` to `useStream`. ⏎ This tells the hook to load the full checkpoint timeline for the current thread. ⏎  ⏎ :::python ⏎  ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type paramete …
```

**chunk 2** (979 chars)

```
```ts ⏎ import type { myAgent } from "./agent"; ⏎ ``` ⏎  ⏎ ::: ⏎  ⏎ <CodeGroup> ⏎ ```tsx React ⏎ import { useStream } from "@langchain/react"; ⏎  ⏎ const AGENT_URL = "http://localhost:2024"; ⏎  ⏎ export function TimeTravelChat() { ⏎   const stream = useStream<typeof myAgent>({ ⏎     apiUrl: AGENT_URL, ⏎     assistantId: "time_travel", ⏎     fetchStateHistory: true, ⏎   }); ⏎  ⏎   const history = stream.history ?? []; ⏎  ⏎   return ( ⏎     <div className="flex h-screen"> ⏎       <ChatPanel messag …
```

**chunk 3** (890 chars)

```
const history = computed(() => stream.history.value ?? []); ⏎  ⏎ function resumeFrom(cp: ThreadState) { ⏎   stream.submit(null, { checkpoint: cp.checkpoint }); ⏎ } ⏎ </script> ⏎  ⏎ <template> ⏎   <div class="flex h-screen"> ⏎     <ChatPanel :messages="stream.messages.value" /> ⏎     <TimelineSidebar :history="history" @select="resumeFrom" /> ⏎   </div> ⏎ </template> ⏎ ``` ⏎  ⏎ ```svelte Svelte ⏎ <script lang="ts"> ⏎   import { useStream } from "@langchain/svelte"; ⏎  ⏎   const AGENT_URL = "http: …
```

### [markdown] — 18 chunks
**chunk 0** (554 chars)

```
Every state change in a LangGraph agent creates a **checkpoint**, a complete ⏎ snapshot of the agent's state at that moment. Time travel lets you inspect any ⏎ checkpoint, view the exact state the agent held, and **resume execution from ⏎ that point** to explore alternative paths. It's a debugger, an undo button, and ⏎ an audit log all in one.   ⏎ import { PatternEmbed } from "/snippets/pattern-embed.jsx"   ⏎ <PatternEmbed pattern="time-travel" />   ⏎ import RequiresLanggraphServer from '/snippe …
```

**chunk 1** (609 chars)

```
## How checkpoints work   ⏎ LangGraph persists agent state after every node execution. Each persisted state ⏎ is a `ThreadState` object that captures:   ⏎ - **checkpoint**: metadata identifying this specific snapshot (ID, timestamp) ⏎ - **values**: the full agent state at this point (messages, custom keys) ⏎ - **tasks**: the graph nodes that were scheduled to run next ⏎ - **next**: the names of upcoming nodes in the execution plan   ⏎ This creates a linear timeline of every decision the agent ma …
```

**chunk 2** (855 chars)

```
## Setting up useStream   ⏎ Enable checkpoint history by passing `fetchStateHistory: true` to `useStream`. ⏎ This tells the hook to load the full checkpoint timeline for the current thread.   ⏎ :::python   ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type parameter to `useStream` for type-safe access to state values. In the examples below, replace `typeof myAgent` with your interface name:   ⏎ ```ts ⏎ import type { BaseMessage } from "@langchain/core/messag …
```

**chunk 3** (886 chars)

```
const AGENT_URL = "http://localhost:2024"; ⏎  ⏎ export function TimeTravelChat() { ⏎ const stream = useStream<typeof myAgent>({ ⏎ apiUrl: AGENT_URL, ⏎ assistantId: "time_travel", ⏎ fetchStateHistory: true, ⏎ }); ⏎  ⏎ const history = stream.history ?? []; ⏎  ⏎ return ( ⏎ <div className="flex h-screen"> ⏎ <ChatPanel messages={stream.messages} /> ⏎ <TimelineSidebar ⏎ history={history} ⏎ onSelect={(cp) => stream.submit(null, { checkpoint: cp.checkpoint })} ⏎ /> ⏎ </div> ⏎ ); ⏎ } ⏎ ```   ⏎ ```vue Vue …
```

### [semantic] — 4 chunks
**chunk 0** (1091 chars)

```
Every state change in a LangGraph agent creates a **checkpoint**, a complete ⏎ snapshot of the agent's state at that moment. Time travel lets you inspect any ⏎ checkpoint, view the exact state the agent held, and **resume execution from ⏎ that point** to explore alternative paths. It's a debugger, an undo button, and ⏎ an audit log all in one. import { PatternEmbed } from "/snippets/pattern-embed.jsx" ⏎  ⏎ <PatternEmbed pattern="time-travel" /> ⏎  ⏎ import RequiresLanggraphServer from '/snippets …
```

**chunk 1** (7697 chars)

```
Your UI can render this timeline and let ⏎ users jump to any point. ## Setting up useStream ⏎  ⏎ Enable checkpoint history by passing `fetchStateHistory: true` to `useStream`. This tells the hook to load the full checkpoint timeline for the current thread. :::python ⏎  ⏎ Define a TypeScript interface matching your agent's state schema and pass it as a type parameter to `useStream` for type-safe access to state values. In the examples below, replace `typeof myAgent` with your interface name: ⏎  ⏎ …
```

**chunk 2** (2294 chars)

```
Re-execute the graph from that point forward ⏎ 3. Stream the new results to the client ⏎  ⏎ The existing messages after the selected checkpoint are replaced by the new ⏎ execution path. This effectively creates a **branch** in the conversation ⏎ timeline. <Note> ⏎ Resuming from a checkpoint does not delete the original timeline. The previous ⏎ checkpoints remain available in the history. This means users can always go back ⏎ and try a different path without losing any prior work. </Note> ⏎  ⏎ ## …
```

**chunk 3** (3180 chars)

```
0, ⏎     hasInterrupts: cp.tasks?.some((t) => t.interrupts?.length) ?? false, ⏎     nextNodes: cp.next ?? [], ⏎   })); ⏎ } ⏎ ``` ⏎  ⏎ This makes it easy to render timeline entries with meaningful labels instead of ⏎ raw IDs. ## Use cases ⏎  ⏎ Time travel is invaluable across many scenarios: ⏎  ⏎ - **Debugging agent behavior**: step through the agent's decisions to ⏎   understand why it chose a particular path ⏎ - **Undoing actions**: if the agent took a wrong turn, resume from an earlier ⏎   che …
```


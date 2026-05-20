import { WorkstationShell } from "@/components/workstation-shell";
import { createAssistantClient } from "@/lib/assistant-client";

export default async function Home() {
  const client = createAssistantClient();
  const snapshot = await client.getWorkspaceSnapshot();

  return <WorkstationShell snapshot={snapshot} />;
}

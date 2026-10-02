import { useChat } from "@/hooks/useChat";
import { AppLayout } from "@/layouts/AppLayout";
import { HomePage } from "@/pages/HomePage";

function App() {
  const chat = useChat();

  return (
    <AppLayout hideFooter={chat.hasStarted} onOpenChat={chat.hasStarted ? undefined : chat.startChat}>
      <HomePage chat={chat} />
    </AppLayout>
  );
}

export default App;

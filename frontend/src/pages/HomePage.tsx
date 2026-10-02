import { DomainCards } from "@/components/home/DomainCards";
import { Hero } from "@/components/home/Hero";
import { OrchestratorSection } from "@/components/home/OrchestratorSection";
import { QuickActions } from "@/components/home/QuickActions";

export function HomePage() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-20 px-4 py-16 sm:gap-24 sm:px-6 sm:py-24 lg:px-8">
      <Hero />
      <QuickActions />
      <DomainCards />
      <OrchestratorSection />
    </div>
  );
}

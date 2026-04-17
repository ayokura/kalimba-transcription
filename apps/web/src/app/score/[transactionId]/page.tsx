import { ScoreViewer } from "@/components/ScoreViewer";

export default async function ScorePage({
  params,
}: {
  params: Promise<{ transactionId: string }>;
}) {
  const { transactionId } = await params;
  return <ScoreViewer transactionId={transactionId} />;
}

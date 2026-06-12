import { ReviewEditor } from "@/components/ReviewEditor";

export default async function ScoreReviewPage({
  params,
}: {
  params: Promise<{ transactionId: string }>;
}) {
  const { transactionId } = await params;
  return <ReviewEditor transactionId={transactionId} />;
}

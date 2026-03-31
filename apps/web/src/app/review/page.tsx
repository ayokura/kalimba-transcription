import { Suspense } from "react";

import { ReviewWorkspace } from "@/components/ReviewWorkspace";

export default function ReviewPage() {
  return (
    <Suspense>
      <ReviewWorkspace />
    </Suspense>
  );
}

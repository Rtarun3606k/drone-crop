import { storeZip } from "@/app/lib/storage";
import { NextResponse } from "next/server";

export async function POST(request) {
  const formData = await request.formData();

  // Get the uploaded ZIP file
  const zipFile = formData.get("imagesZip");
  const batchName = formData.get("batchName");
  const cropType = formData.get("cropType");

  if (!zipFile || !zipFile.name.endsWith(".zip")) {
    return NextResponse.json(
      { error: "ZIP file missing or invalid." },
      { status: 400 }
    );
  }

  const uploadToBucket = await storeZip(zipFile);

  if (!uploadToBucket.success) {
    return NextResponse.json(
      { error: "Failed to store ZIP file." },
      { status: 500 }
    );
  }

  console.log("📦 ZIP file received:", zipFile.name);
  console.log(
    "📦 ZIP file received:",
    (zipFile.size / (1024 * 1024)).toFixed(2) + " MB"
  );
  console.log("📂 Batch Name:", batchName);
  console.log("🌱 Crop Type:", cropType);

  // Read ZIP content (Blob → Buffer)
  const arrayBuffer = await zipFile.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  // You can now extract buffer using unzipper or adm-zip
  // e.g., await unzip(buffer) or save to temp folder

  return NextResponse.json({ message: "Upload received!" }, { status: 200 });
}

import { auth } from "@/app/auth";
import { prisma } from "@/app/lib/prisma-server";
import { storeZip } from "@/app/lib/storage";
import { NextResponse } from "next/server";
import { data } from "react-router-dom";

export async function POST(request) {
  const session = await auth();
  const formData = await request.formData();
  if (!session?.user || session === null) {
    return NextResponse.json(
      { error: "Unauthorized. Please log in." },
      { status: 401 }
    );
  }

  // Get the uploaded ZIP file
  const zipFile = formData.get("imagesZip");
  const batchName = formData.get("batchName");
  const cropType = formData.get("cropType");
  const imagesCount = formData.get("imagesCount");

  if (!zipFile || !zipFile.name.endsWith(".zip")) {
    return NextResponse.json(
      { error: "ZIP file missing or invalid." },
      { status: 400 }
    );
  }

  const uploadToBucket = await storeZip(zipFile, zipFile.name);

  if (!uploadToBucket.success) {
    return NextResponse.json(
      { error: "Failed to store ZIP file." },
      { status: 500 }
    );
  }
  const DB = await prisma.batch.create({
    data: {
      name: batchName,
      cropType: cropType,
      imagesZipURL: uploadToBucket.url || `shared/${zipFile.name}`,
      imagesCount: parseInt(imagesCount) || 0,
      userId: session.user.id, // You need to provide the user ID
      // The following fields have defaults, so you don't need to provide them:
      // createdAt: new Date(), // This has @default(now())
      // isCompletedModel: false, // This has @default(false)
      // isCompleted: false, // This has @default(false)
      // sessionId will get a UUID automatically
    },
  });
  if (!DB) {
    return NextResponse.json(
      { error: "Failed to save batch data." },
      { status: 500 }
    );
  }
  console.log("✅ Batch data saved to database:", DB);
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

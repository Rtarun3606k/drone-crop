import { auth } from "@/app/auth";
import { prisma } from "@/app/lib/prisma-server";
import { NextResponse } from "next/server";

export async function GET(request, { params }) {
  try {
    const session = await auth();

    // Check authentication
    if (!session?.user) {
      return NextResponse.json(
        { error: "Unauthorized. Please log in." },
        { status: 401 }
      );
    }

    const { id } = await params;

    // Validate the ID format for MongoDB ObjectId
    if (!id.match(/^[0-9a-fA-F]{24}$/)) {
      return NextResponse.json(
        { error: "Invalid batch ID format." },
        { status: 400 }
      );
    }

    // Fetch the specific batch with related audioFiles and descriptions
    const batch = await prisma.batch.findUnique({
      where: {
        id: id,
        userId: session.user.id,
      },
      include: {
        audioFiles: true,
        descriptions: true, // ✅ Fetch descriptions directly here
      },
    });

    if (!batch) {
      return NextResponse.json(
        { error: "Batch not found or you don't have permission to view it." },
        { status: 404 }
      );
    }

    return NextResponse.json({ batch }, { status: 200 });
  } catch (error) {
    console.error("Error fetching batch details:", error);
    return NextResponse.json(
      { error: "Failed to fetch batch details." },
      { status: 500 }
    );
  }
}

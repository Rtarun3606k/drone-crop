import { auth } from "@/app/auth";
import { prisma } from "@/app/lib/prisma-server";
import { NextResponse } from "next/server";

/**
 * @swagger
 * /api/batches/{id}:
 *   get:
 *     summary: Get a specific batch by ID with audio files and descriptions
 *     tags:
 *       - Batches
 *     parameters:
 *       - name: id
 *         in: path
 *         required: true
 *         schema:
 *           type: string
 *           pattern: "^[0-9a-fA-F]{24}$"
 *         description: The ID of the batch (must be a valid 24-character hex string)
 *     security:
 *       - cookieAuth: []
 *     responses:
 *       200:
 *         description: Batch details with audio files and descriptions
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 batch:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     name:
 *                       type: string
 *                     createdAt:
 *                       type: string
 *                       format: date-time
 *                     audioFiles:
 *                       type: array
 *                       items:
 *                         type: object
 *                     descriptions:
 *                       type: array
 *                       items:
 *                         type: object
 *       400:
 *         description: Invalid batch ID format
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Batch not found or access denied
 *       500:
 *         description: Server error
 */

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

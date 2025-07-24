import { auth } from "@/app/auth";
import { prisma } from "@/app/lib/prisma-server";
import { NextResponse } from "next/server";

/**
 * @swagger
 * /api/batches:
 *   get:
 *     summary: Get all batches for the authenticated user
 *     tags:
 *       - Batches
 *     security:
 *       - cookieAuth: []
 *     responses:
 *       200:
 *         description: A list of batches
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 batches:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       name:
 *                         type: string
 *                       createdAt:
 *                         type: string
 *                         format: date-time
 *       401:
 *         description: Unauthorized. Please log in.
 *       500:
 *         description: Failed to fetch batches
 */

export async function GET(request) {
  try {
    const session = await auth();

    // Check authentication
    if (!session?.user) {
      return NextResponse.json(
        { error: "Unauthorized. Please log in." },
        { status: 401 }
      );
    }

    // Fetch all batches for the current user
    const batches = await prisma.batch.findMany({
      where: {
        userId: session.user.id,
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return NextResponse.json({ batches }, { status: 200 });
  } catch (error) {
    console.error("Error fetching batches:", error);
    return NextResponse.json(
      { error: "Failed to fetch batches." },
      { status: 500 }
    );
  }
}

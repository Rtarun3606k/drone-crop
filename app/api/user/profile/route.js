import { auth } from "@/app/auth";
import { getUserProfile, updateUserProfile } from "@/app/lib/db";
import { NextResponse } from "next/server";
import { authConfig } from "@/app/non-edge-config";

// Force Node.js runtime for this route
export const runtime = authConfig.runtime;
export const preferredRegion = authConfig.preferredRegion;

export async function GET() {
  const session = await auth();

  // Check if the user is authenticated
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const userProfile = await getUserProfile(session.user.id);
    return NextResponse.json(userProfile);
  } catch (error) {
    console.error("Error fetching user profile:", error);
    return NextResponse.json(
      { error: "Failed to fetch user profile" },
      { status: 500 }
    );
  }
}

export async function POST(request) {
  const session = await auth();

  // Check if the user is authenticated
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const args = await request.json();

    // If it's a client-side request to find a user
    if (args.where?.email) {
      const userProfile = await getUserProfile(session.user.id);
      if (userProfile.email !== args.where.email) {
        // Security check - only allow querying own email
        return NextResponse.json({ error: "Forbidden" }, { status: 403 });
      }
      return NextResponse.json(userProfile);
    }

    // If it's an update operation
    if (args.data) {
      const updatedProfile = await updateUserProfile(
        session.user.id,
        args.data
      );
      return NextResponse.json(updatedProfile);
    }

    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  } catch (error) {
    console.error("Error with profile request:", error);
    return NextResponse.json(
      { error: "Failed to process profile request" },
      { status: 500 }
    );
  }
}

/**
 * @swagger
 * /user/profile:
 *   get:
 *     summary: Get the authenticated user's profile
 *     description: Retrieves the profile of the currently authenticated user.
 *     tags:
 *       - User
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Successful retrieval of user profile
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                 name:
 *                   type: string
 *                 email:
 *                   type: string
 *                 metadata:
 *                   type: object
 *       401:
 *         description: Unauthorized - User not authenticated
 *       500:
 *         description: Internal Server Error
 *
 *   post:
 *     summary: Get or update the authenticated user's profile
 *     description: >
 *       If `where.email` is provided, returns the user profile for that email (only if it matches the authenticated user).
 *       If `data` is provided, updates the user's profile.
 *     tags:
 *       - User
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             oneOf:
 *               - type: object
 *                 properties:
 *                   where:
 *                     type: object
 *                     properties:
 *                       email:
 *                         type: string
 *               - type: object
 *                 properties:
 *                   data:
 *                     type: object
 *                     properties:
 *                       metadata:
 *                         type: object
 *     responses:
 *       200:
 *         description: Successful operation (fetch or update)
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                 name:
 *                   type: string
 *                 email:
 *                   type: string
 *                 metadata:
 *                   type: object
 *       401:
 *         description: Unauthorized - User not authenticated
 *       403:
 *         description: Forbidden - Trying to access another user's email
 *       400:
 *         description: Invalid request format
 *       500:
 *         description: Internal Server Error
 */

import { auth } from "@/app/auth";
import {
  getUserSessions,
  deleteSession,
  deleteOtherSessions,
} from "@/app/lib/db";
import { NextResponse } from "next/server";
import { authConfig } from "@/app/non-edge-config";

// Force Node.js runtime for this route
export const runtime = authConfig.runtime;
export const preferredRegion = authConfig.preferredRegion;

export async function GET(request) {
  const session = await auth();

  // Check if the user is authenticated
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const userSessions = await getUserSessions(session.user.id);

    // Format the sessions for safe client-side consumption
    const formattedSessions = userSessions.map((s) => ({
      id: s.id,
      sessionToken: s.sessionToken,
      expires: s.expires,
      lastUsed: s.expires, // Using expires as an approximation
      current: session.sessionToken === s.sessionToken,
    }));

    // Include user role in the response
    return NextResponse.json({
      sessions: formattedSessions,
      userRole: session.user.role,
    });
  } catch (error) {
    console.error("Error fetching sessions:", error);
    return NextResponse.json(
      { error: "Failed to fetch sessions" },
      { status: 500 }
    );
  }
}

// Delete a specific session
export async function DELETE(request) {
  const session = await auth();

  // Check if the user is authenticated
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const { searchParams } = new URL(request.url);
    const sessionToken = searchParams.get("sessionToken");
    const terminateAll = searchParams.get("all") === "true";

    if (!sessionToken && !terminateAll) {
      return NextResponse.json(
        { error: "Session token is required" },
        { status: 400 }
      );
    }

    // Don't allow deleting the current session directly
    if (sessionToken === session.sessionToken) {
      return NextResponse.json(
        { error: "Cannot delete current session" },
        { status: 400 }
      );
    }

    if (terminateAll) {
      // Delete all other sessions
      await deleteOtherSessions(session.user.id, session.sessionToken);
      return NextResponse.json({
        success: true,
        message: "All other sessions terminated",
      });
    } else {
      // Delete specific session
      await deleteSession(sessionToken);
      return NextResponse.json({
        success: true,
        message: "Session terminated",
      });
    }
  } catch (error) {
    console.error("Error deleting session:", error);
    return NextResponse.json(
      { error: "Failed to delete session" },
      { status: 500 }
    );
  }
}

/**
 * @swagger
 * /user/profile:
 *   get:
 *     summary: Get all active sessions and user role
 *     description: Returns a list of active sessions for the authenticated user, including the user's role.
 *     tags:
 *       - User
 *     responses:
 *       200:
 *         description: Successfully retrieved sessions
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 sessions:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       sessionToken:
 *                         type: string
 *                       expires:
 *                         type: string
 *                         format: date-time
 *                       lastUsed:
 *                         type: string
 *                         format: date-time
 *                       current:
 *                         type: boolean
 *                 userRole:
 *                   type: string
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Failed to fetch sessions
 *
 *   delete:
 *     summary: Terminate a user session
 *     description: Delete a specific session or all sessions except the current one for the authenticated user.
 *     tags:
 *       - User
 *     parameters:
 *       - in: query
 *         name: sessionToken
 *         schema:
 *           type: string
 *         required: false
 *         description: Token of the session to delete
 *       - in: query
 *         name: all
 *         schema:
 *           type: boolean
 *         required: false
 *         description: Whether to delete all other sessions
 *     responses:
 *       200:
 *         description: Successfully deleted session(s)
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *       400:
 *         description: Invalid request or cannot delete current session
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Failed to delete session
 */

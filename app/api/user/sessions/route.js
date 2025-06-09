import { auth } from "@/app/auth";
import {
  getUserSessions,
  deleteSession,
  deleteOtherSessions,
} from "@/app/lib/db";
import { NextResponse } from "next/server";

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

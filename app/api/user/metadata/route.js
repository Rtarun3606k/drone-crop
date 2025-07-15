const { auth } = require("@/app/auth");
const { NextResponse } = require("next/server");

const updateUserProfile = async (request) => {
  const session = await auth();
  if (!session || !session.user) {
    throw new Error("Unauthorized");
  }

  try {
    const data = request.json();
    const parsedMetadata = data.metadata ? JSON.parse(data.metadata) : {};
    const updatedProfile = await prisma.user.update({
      where: { id: session.user.id },
      data: {
        metadata: data.metadata || {},
      },
    });

    return NextResponse.json(
      { success: true, profile: updatedProfile },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error updating user profile:", error);
    return NextResponse.json(
      { error: "Failed to update user profile" },
      { status: 500 }
    );
  }
  // Logic to update user profile
  // This is a placeholder; actual implementation will depend on your database and requirements
};

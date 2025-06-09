// A simple script to set a user as an admin
const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient();

async function setUserAdmin(email) {
  try {
    const user = await prisma.user.update({
      where: { email },
      data: { role: "ADMIN" },
    });

    console.log(`✅ User ${email} has been set as ADMIN`);
    console.log(user);
    return user;
  } catch (error) {
    console.error("❌ Error setting user as admin:", error);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
}

// Get email from command line arguments
const email = process.argv[2];

if (!email) {
  console.error("❌ Please provide a user email as an argument");
  console.log("Usage: node scripts/set-admin.js user@example.com");
  process.exit(1);
}

setUserAdmin(email)
  .then(() => process.exit(0))
  .catch(() => process.exit(1));

import { app, prisma } from "./config.mjs";
import authRoutes from "./Routes/auth.mjs";
import userRoutes from "./Routes/user.mjs"; // Assuming you have user routes

// Basic route
app.get("/", async (req, res) => {
  const message = await prisma.user.findMany();
  //   console.log(message);
  res.json({
    message: "Welcome to the Drone Crop Mobile Backend API",
    timestamp: new Date().toISOString(),
    messageFromDB: message,
  });
});

// Importing routes
app.use("/api/auth", authRoutes);
app.use("/api/user", userRoutes);

// 404 handler - catch all remaining routes
app.use((req, res) => {
  res.status(404).json({
    error: "Endpoint not found",
    message: `The requested endpoint ${req.originalUrl} does not exist.`,
  });
});

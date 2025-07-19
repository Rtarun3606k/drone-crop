import { Router } from "express";
import { prisma } from "../config.mjs";

import { createJwtToken, refreshTokenIfNeeded } from "../lib/jwtTokens.mjs";
import { verifyToken } from "../middleware/midddleware.mjs";

const router = Router();

router.get("/get-user", verifyToken, refreshTokenIfNeeded, async (req, res) => {
  const userEmail = req.user.email;
  const user = await prisma.user.findUnique({
    where: {
      email: userEmail,
    },
  });

  if (!user) {
    return res.status(404).json({
      error: "User not found",
      message: "No user found with the provided email.",
    });
  }

  res.status(200).json({
    message: "User retrieved successfully",
    user: user,
  });
});

export default router;

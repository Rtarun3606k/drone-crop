import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";

export const storeZip = async (file, name) => {
  try {
    // Ensure the public/shared directory exists in the parent directory
    const uploadDir = "/home/dragoon/coding/drone-crop/public/shared";
    if (!existsSync(uploadDir)) {
      await mkdir(uploadDir, { recursive: true });
    }

    // Create unique filename with timestamp
    const fileName = `${name.replace(/\.zip$/, "")}-${Date.now()}.zip`;
    const filePath = join(uploadDir, fileName);

    // Write the file buffer to disk
    await writeFile(filePath, file.buffer);

    return {
      success: true,
      path: filePath,
      url: `public/shared/${fileName}`,
    };
  } catch (error) {
    console.error("Error storing ZIP file:", error);
    return {
      success: false,
      error: "Failed to store ZIP file",
    };
  }
};

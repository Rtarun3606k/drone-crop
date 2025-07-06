import { writeFile } from "fs/promises";
import { join } from "path";

export const storeZip = async (zip, name) => {
  try {
    const arrayBuffer = await zip.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const fileName = `${zip.name.replace(/\.zip$/, "")}-${Date.now()}.zip`;
    const filePath = join(process.cwd(), "shared", name);

    await writeFile(filePath, buffer);
    return { success: true, path: filePath };
  } catch (error) {
    console.error("Error storing ZIP file:", error);
    return { success: false, error: "Failed to store ZIP file" };
  }
};

import { app, PORT } from "./config.mjs";
import "./app.js"; // Import routes

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server is running on port ${PORT}`);
});

import { app, PORT } from "./config.mjs";
import "./app.js"; // Import routes

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

const express = require("express");
const mongoose = require("mongoose");
const path = require("path");
const port = 3018;

const app = express();

// Middleware
app.use(express.static(__dirname));  // serve static files from current folder
app.use(express.urlencoded({ extended: true }));
app.use(express.json()); // For parsing JSON requests

// Connect to MongoDB
mongoose.connect("mongodb://127.0.0.1:27017/thyronet");
const db = mongoose.connection;
db.once("open", () => {
  console.log("Mongoose Connection Successful");
});

// Mongoose schema and model
const userSchema = new mongoose.Schema({
  name: String,
  email: { type: String, unique: true },
  password: String,
});

const Users = mongoose.model("LoerDetails", userSchema);

// Serve Login Page
app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, 'form.html'));
});

// Handle Login
app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await Users.findOne({ email });

    if (!user) {
      return res.json({
        success: false,
        message: "User not registered! Please register first.",
      });
    }

    if (user.password !== password) {
      return res.json({ success: false, message: "Invalid email or password!" });
    }

    // ✅ Redirect to Flask app
    res.json({ success: true, redirect: "http://127.0.0.1:5000" });
  } catch (err) {
    console.error("Error during login:", err);
    res.json({
      success: false,
      message: "An error occurred during login. Please try again later.",
    });
  }
});

// Serve Registration Page
app.get("/register", (req, res) => {
  res.sendFile(path.join(__dirname, "register.html"));
});

// Handle Registration
app.post("/register", async (req, res) => {
  const { name, email, password } = req.body;

  try {
    const userExists = await Users.findOne({ email });
    if (userExists) {
      return res.send(`
                <!DOCTYPE html>
                <html>
                <head><title>Error</title></head>
                <body>
                    <script>
                        alert("Error: User already exists. Please log in instead.");
                        window.location.href = "/login";
                    </script>
                </body>
                </html>
            `);
    }

    const user = new Users({ name, email, password });
    await user.save();
    res.send(`
            <!DOCTYPE html>
            <html>
            <head><title>Success</title></head>
            <body>
                <script>
                    alert("Registration successful!");
                    window.location.href = "/login";
                </script>
            </body>
            </html>
        `);
  } catch (err) {
    console.error(err);
    res.send(`
            <!DOCTYPE html>
            <html>
            <head><title>Error</title></head>
            <body>
                <script>
                    alert("An error occurred during registration. Please try again.");
                    window.location.href = "/register";
                </script>
            </body>
            </html>
        `);
  }
});

// Start Server
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});

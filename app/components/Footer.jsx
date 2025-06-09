import React from "react";
import Link from "next/link";

const Footer = () => (
  <footer style={{
    background: "#222",
    color: "#fff",
    padding: "2rem 0",
    textAlign: "center",
    marginTop: "auto"
  }}>
    <div style={{ marginBottom: "1rem" }}>
      <h2 style={{ margin: 0, fontSize: "1.5rem" }}>DroneCrop</h2>
    </div>
    <nav style={{ marginBottom: "1rem" }}>
      <Link href="/" style={{ color: "#fff", margin: "0 1rem", textDecoration: "none" }}>Home</Link>
      <Link href="/profile" style={{ color: "#fff", margin: "0 1rem", textDecoration: "none" }}>Profile</Link>
      <Link href="/services" style={{ color: "#fff", margin: "0 1rem", textDecoration: "none" }}>Services</Link>
      <Link href="/about" style={{ color: "#fff", margin: "0 1rem", textDecoration: "none" }}>About</Link>
      <Link href="/contact" style={{ color: "#fff", margin: "0 1rem", textDecoration: "none" }}>Contact</Link>
    </nav>
    <div>
      <p style={{ margin: 0 }}>
        Contact us: <a href="mailto:info@dronecrop.com" style={{ color: "#4caf50" }}>info@dronecrop.com</a> | +1 (555) 123-4567
      </p>
      <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.9rem", color: "#aaa" }}>
        &copy; {new Date().getFullYear()} DroneCrop. All rights reserved.
      </p>
    </div>
  </footer>
);

export default Footer;
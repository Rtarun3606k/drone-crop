"use client";
import React, { useState } from "react";
import { signIn, useSession } from "next-auth/react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle login logic here
  };

  const handleGoogleSignIn = async () => {
    setIsLoading(true);
    try {
      await signIn("google", { callbackUrl: "/" });
    } catch (error) {
      console.error("Sign-in error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #232526 0%, #414345 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "Inter, sans-serif",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          background: "rgba(255,255,255,0.05)",
          borderRadius: "20px",
          boxShadow: "0 8px 32px 0 rgba(31,38,135,0.37)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(255,255,255,0.18)",
          padding: "40px 32px",
          width: "350px",
          display: "flex",
          flexDirection: "column",
          gap: "24px",
        }}
      >
        <h2
          style={{
            color: "#fff",
            textAlign: "center",
            marginBottom: "8px",
            letterSpacing: "1px",
          }}
        >
          Welcome Back
        </h2>
        <p
          style={{
            color: "#aaa",
            textAlign: "center",
            marginBottom: "16px",
            fontSize: "15px",
          }}
        >
          Sign in to your account
        </p>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          style={{
            padding: "12px",
            borderRadius: "8px",
            border: "none",
            outline: "none",
            background: "rgba(255,255,255,0.15)",
            color: "#fff",
            fontSize: "16px",
          }}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={{
            padding: "12px",
            borderRadius: "8px",
            border: "none",
            outline: "none",
            background: "rgba(255,255,255,0.15)",
            color: "#fff",
            fontSize: "16px",
          }}
        />
        <button
          type="submit"
          style={{
            padding: "12px",
            borderRadius: "8px",
            border: "none",
            background: "linear-gradient(90deg, #ff512f 0%, #dd2476 100%)",
            color: "#fff",
            fontWeight: "bold",
            fontSize: "16px",
            cursor: "pointer",
            transition: "background 0.2s",
          }}
        >
          Log In
        </button>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginTop: "16px",
            marginBottom: "16px",
          }}
        >
          <div
            style={{
              flex: 1,
              height: "1px",
              background: "rgba(255,255,255,0.1)",
            }}
          ></div>
          <span style={{ padding: "0 10px", color: "#aaa", fontSize: "14px" }}>
            OR
          </span>
          <div
            style={{
              flex: 1,
              height: "1px",
              background: "rgba(255,255,255,0.1)",
            }}
          ></div>
        </div>
        <button
          type="button"
          onClick={handleGoogleSignIn}
          disabled={isLoading}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "10px",
            padding: "12px",
            borderRadius: "8px",
            border: "1px solid rgba(255,255,255,0.2)",
            background: "rgba(255,255,255,0.1)",
            color: "#fff",
            fontWeight: "bold",
            fontSize: "16px",
            cursor: "pointer",
            transition: "background 0.2s",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 48 48"
          >
            <path
              fill="#FFC107"
              d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"
            />
            <path
              fill="#FF3D00"
              d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"
            />
            <path
              fill="#4CAF50"
              d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"
            />
            <path
              fill="#1976D2"
              d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"
            />
          </svg>
          {isLoading ? "Loading..." : "Sign in with Google"}
        </button>
        <div style={{ textAlign: "center", marginTop: "16px" }}>
          <a
            href="#"
            style={{
              color: "#ff7eb3",
              textDecoration: "none",
              fontSize: "14px",
            }}
          >
            Forgot password?
          </a>
        </div>
        <div style={{ textAlign: "center", marginTop: "8px" }}>
          <a
            href="/register"
            style={{
              color: "#7ee8fa",
              textDecoration: "none",
              fontSize: "14px",
            }}
          >
            New user? Register here
          </a>
        </div>
      </form>
    </div>
  );
}

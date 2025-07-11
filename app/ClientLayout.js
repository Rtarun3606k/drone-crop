"use client";

// import Navbar from "./components/Navbar";

export default function ClientLayout({ children }) {
  return (
    <>
      {/* <Navbar /> */}
      <div className="pt-16">{children}</div>
    </>
  );
}

"use client";
import React, { useState } from 'react'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    // Handle login logic here
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #232526 0%, #414345 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Inter, sans-serif'
    }}>
      <form
        onSubmit={handleSubmit}
        style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '20px',
          boxShadow: '0 8px 32px 0 rgba(31,38,135,0.37)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.18)',
          padding: '40px 32px',
          width: '350px',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px'
        }}
      >
        <h2 style={{
          color: '#fff',
          textAlign: 'center',
          marginBottom: '8px',
          letterSpacing: '1px'
        }}>
          Welcome Back
        </h2>
        <p style={{
          color: '#aaa',
          textAlign: 'center',
          marginBottom: '16px',
          fontSize: '15px'
        }}>
          Sign in to your account
        </p>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          style={{
            padding: '12px',
            borderRadius: '8px',
            border: 'none',
            outline: 'none',
            background: 'rgba(255,255,255,0.15)',
            color: '#fff',
            fontSize: '16px'
          }}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          style={{
            padding: '12px',
            borderRadius: '8px',
            border: 'none',
            outline: 'none',
            background: 'rgba(255,255,255,0.15)',
            color: '#fff',
            fontSize: '16px'
          }}
        />
        <button
          type="submit"
          style={{
            padding: '12px',
            borderRadius: '8px',
            border: 'none',
            background: 'linear-gradient(90deg, #ff512f 0%, #dd2476 100%)',
            color: '#fff',
            fontWeight: 'bold',
            fontSize: '16px',
            cursor: 'pointer',
            transition: 'background 0.2s'
          }}
        >
          Log In
        </button>
        <div style={{ textAlign: 'center', marginTop: '8px' }}>
          <a href="#" style={{ color: '#ff7eb3', textDecoration: 'none', fontSize: '14px' }}>
            Forgot password?
          </a>
        </div>
        <div style={{ textAlign: 'center', marginTop: '8px' }}>
          <a href="/register" style={{ color: '#7ee8fa', textDecoration: 'none', fontSize: '14px' }}>
            New user? Register here
          </a>
        </div>
      </form>
    </div>
  )
}

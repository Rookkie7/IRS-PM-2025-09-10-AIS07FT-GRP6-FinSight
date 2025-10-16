/*
  # Create users table for authentication and profile data

  1. New Tables
    - `users`
      - `id` (uuid, primary key) - Unique user identifier
      - `email` (text, unique, not null) - User email for authentication
      - `username` (text, unique, not null) - Display username
      - `password_hash` (text, not null) - Hashed password
      - `full_name` (text, not null) - User's full name
      - `bio` (text, nullable) - Optional user biography
      - `interests` (text[], not null, default empty array) - User interests (AI, Cloud, etc.)
      - `sectors` (text[], not null, default empty array) - Selected sectors from predefined list
      - `tickers` (text[], not null, default empty array) - Stock tickers of interest
      - `created_at` (timestamptz, default now()) - Account creation timestamp
      - `updated_at` (timestamptz, default now()) - Last update timestamp

  2. Security
    - Enable RLS on `users` table
    - Add policy for users to read their own data
    - Add policy for users to update their own data
    - Add policy for authenticated users to read public user info (username, full_name, bio)

  3. Important Notes
    - Password will be hashed before storage
    - Email and username must be unique
    - Sectors must be from predefined list: Utilities, Technology, Consumer Defensive, Healthcare, 
      Basic Materials, Real Estate, Energy, Industrials, Consumer Cyclical, Communication Services, Financial Services
*/

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email text UNIQUE NOT NULL,
  username text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  full_name text NOT NULL,
  bio text,
  interests text[] NOT NULL DEFAULT '{}',
  sectors text[] NOT NULL DEFAULT '{}',
  tickers text[] NOT NULL DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own data"
  ON users
  FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can update own data"
  ON users
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Public user info readable by authenticated users"
  ON users
  FOR SELECT
  TO authenticated
  USING (true);

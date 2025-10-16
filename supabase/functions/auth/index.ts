import { createClient } from 'npm:@supabase/supabase-js@2.57.4';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Client-Info, Apikey',
};

interface RegisterPayload {
  email: string;
  username: string;
  password: string;
  full_name: string;
  bio?: string;
  interests: string[];
  sectors: string[];
  tickers: string[];
}

interface LoginPayload {
  email: string;
  password: string;
}

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const url = new URL(req.url);
    const path = url.pathname;

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Register endpoint
    if (path.endsWith('/register') && req.method === 'POST') {
      const { payload }: { payload: RegisterPayload } = await req.json();

      // Validate required fields
      if (!payload.email || !payload.username || !payload.password || !payload.full_name) {
        return new Response(
          JSON.stringify({ error: 'Missing required fields' }),
          {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      // Check if user already exists
      const { data: existingUser } = await supabase
        .from('users')
        .select('id')
        .or(`email.eq.${payload.email},username.eq.${payload.username}`)
        .maybeSingle();

      if (existingUser) {
        return new Response(
          JSON.stringify({ error: 'User with this email or username already exists' }),
          {
            status: 409,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      // Hash password using Web Crypto API
      const encoder = new TextEncoder();
      const data = encoder.encode(payload.password);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const passwordHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

      // Insert new user
      const { data: newUser, error: insertError } = await supabase
        .from('users')
        .insert({
          email: payload.email,
          username: payload.username,
          password_hash: passwordHash,
          full_name: payload.full_name,
          bio: payload.bio || null,
          interests: payload.interests || [],
          sectors: payload.sectors || [],
          tickers: payload.tickers || [],
        })
        .select('id, email, username, full_name, bio, interests, sectors, tickers, created_at')
        .single();

      if (insertError) {
        return new Response(
          JSON.stringify({ error: 'Failed to create user', details: insertError.message }),
          {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      // Generate a simple token (in production, use JWT)
      const token = btoa(`${newUser.id}:${Date.now()}`);

      return new Response(
        JSON.stringify({
          message: 'User registered successfully',
          token,
          user: newUser,
        }),
        {
          status: 201,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // Login endpoint
    if (path.endsWith('/login') && req.method === 'POST') {
      const { email, password }: LoginPayload = await req.json();

      if (!email || !password) {
        return new Response(
          JSON.stringify({ error: 'Email and password are required' }),
          {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      // Hash the provided password
      const encoder = new TextEncoder();
      const data = encoder.encode(password);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const passwordHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

      // Find user with matching email and password
      const { data: user, error: selectError } = await supabase
        .from('users')
        .select('id, email, username, full_name, bio, interests, sectors, tickers, created_at')
        .eq('email', email)
        .eq('password_hash', passwordHash)
        .maybeSingle();

      if (selectError || !user) {
        return new Response(
          JSON.stringify({ error: 'Invalid email or password' }),
          {
            status: 401,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      // Generate a simple token (in production, use JWT)
      const token = btoa(`${user.id}:${Date.now()}`);

      return new Response(
        JSON.stringify({
          message: 'Login successful',
          token,
          user,
        }),
        {
          status: 200,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    return new Response(
      JSON.stringify({ error: 'Not found' }),
      {
        status: 404,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Internal server error', details: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
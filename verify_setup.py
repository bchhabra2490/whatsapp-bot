#!/usr/bin/env python3
"""
Setup verification script
Checks if all required environment variables and dependencies are configured
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_env_var(name: str, required: bool = True) -> bool:
    """Check if environment variable is set"""
    value = os.getenv(name)
    if required and not value:
        print(f"❌ {name} is not set")
        return False
    elif value:
        print(f"✅ {name} is set")
        return True
    else:
        print(f"⚠️  {name} is not set (optional)")
        return True


def check_imports():
    """Check if all required packages can be imported"""
    packages = [
        ("flask", "Flask"),
        ("twilio", "Twilio"),
        ("supabase", "Supabase"),
        ("requests", "Requests"),
    ]

    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is not installed")
            all_ok = False

    # Check pdf2image (includes Pillow as dependency)
    try:
        import pdf2image
        from PIL import Image

        print(f"✅ pdf2image is installed (PDF support enabled)")
        print(f"✅ Pillow is installed (via pdf2image dependency)")
    except ImportError:
        print(f"⚠️  pdf2image is not installed (PDF support disabled)")

    return all_ok


def main():
    print("🔍 Verifying WhatsApp Bot Setup...\n")

    print("📋 Checking environment variables:")
    env_vars_ok = True
    env_vars_ok &= check_env_var("TWILIO_ACCOUNT_SID", required=True)
    env_vars_ok &= check_env_var("TWILIO_AUTH_TOKEN", required=True)
    env_vars_ok &= check_env_var("TWILIO_WHATSAPP_NUMBER", required=True)
    env_vars_ok &= check_env_var("SUPABASE_URL", required=True)
    env_vars_ok &= check_env_var("SUPABASE_KEY", required=True)
    env_vars_ok &= check_env_var("SUPABASE_STORAGE_BUCKET", required=False)
    env_vars_ok &= check_env_var("MISTRAL_API_KEY", required=True)
    env_vars_ok &= check_env_var("MISTRAL_MODEL", required=False)

    print("\n📦 Checking Python packages:")
    imports_ok = check_imports()

    print("\n" + "=" * 50)
    if env_vars_ok and imports_ok:
        print("✅ Setup verification passed!")
        print("\nNext steps:")
        print("1. Run the database schema: database/schema.sql in Supabase SQL Editor")
        print("2. Create a storage bucket named 'receipts' in Supabase")
        print("3. Configure Twilio webhook: https://your-domain.com/webhook")
        print("4. Run: python app.py")
        return 0
    else:
        print("❌ Setup verification failed!")
        print("\nPlease fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

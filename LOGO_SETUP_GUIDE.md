# Logo Setup Guide for Users

## Quick Start: Adding Your Bank Logo

### Step 1: Create the Logo Directory
Create a folder for your bank's logo (replace `scotiabank` with your bank name):

```bash
mkdir -p src/web/static/images/banks/scotiabank/
```

### Step 2: Add Your Logo Files
Place your logo files in the directory you just created:

```
src/web/static/images/banks/scotiabank/
├── logo.svg    # Your bank logo (recommended)
├── logo.png    # Alternative format
└── favicon.svg # Your bank favicon (optional)
```

### Step 3: That's It!
Your logo will automatically appear on the landing page in the light grey section on the left.

## Supported Logo Formats

The system supports multiple image formats:
- **SVG** (recommended) - Best quality, scales perfectly
- **PNG** - Good quality, supports transparency
- **JPG/JPEG** - Standard format
- **GIF** - Animated or static

## Logo Requirements

- **Size**: 120x120px or larger (will be auto-scaled)
- **Background**: Transparent or white recommended
- **Quality**: High resolution for best display

## Available Banks

Currently supported preset banks:
- **scotiabank** - Scotiabank (Canada)

More banks will be added as presets. You only need to add logos for the banks you use.

## Troubleshooting

### Logo Not Showing?
1. Check the folder name matches exactly: `scotiabank` (not `Scotiabank` or `SCOTIABANK`)
2. Verify your logo file is named `logo.svg` (or other supported format)
3. Make sure the file is readable

### Logo Looks Wrong?
- Try using SVG format for best quality
- Ensure your logo is at least 120x120px
- Check that the background works well with the light grey section

## Example

For Scotiabank users:
1. Create: `src/web/static/images/banks/scotiabank/`
2. Add: `logo.svg` (your Scotiabank logo)
3. Visit: `http://localhost:8000` to see your logo!

That's all you need to do! 🎉 
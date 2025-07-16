# Bank Configuration and Logo Setup

This directory contains **preset bank configurations** that are ready to use. Users do not need to configure banks themselves.

## For Users: Adding Your Bank Statements and Logos

### 1. Create Bank Folder Structure

Simply create a folder in `bank_statements/` with the bank name that matches a preset configuration:

```
bank_statements/
└── scotiabank/          # Bank name must match a preset config
    ├── Credit/
    │   └── YourAccountName/
    │       └── PDF files
    └── Chequing/
        └── YourAccountName/
            └── PDF files
```

**Available Preset Banks:**
- `scotiabank` - Scotiabank (Canada)
- More banks will be added as presets

### 2. Add Your Bank Logo (Optional)

To display your bank's logo on the landing page, add logo files to the corresponding directory:

```bash
# Create the logo directory for your bank
mkdir -p ../../web/static/images/banks/scotiabank/
```

#### Supported Logo Formats
The system supports multiple image formats with automatic fallback:
- **SVG** (recommended) - `/static/images/banks/scotiabank/logo.svg`
- **PNG** - `/static/images/banks/scotiabank/logo.png`
- **JPG/JPEG** - `/static/images/banks/scotiabank/logo.jpg`
- **GIF** - `/static/images/banks/scotiabank/logo.gif`

#### Logo Requirements
- **Size**: Recommended 120x120px or larger (will be scaled down)
- **Format**: SVG preferred for crisp display at any size
- **Background**: Transparent or white background recommended
- **File naming**: Use `logo.svg` (or other extension) as the filename

### 3. Add Favicon (Optional)

Add a favicon for your bank:
- **SVG** (recommended) - `/static/images/banks/scotiabank/favicon.svg`
- **ICO** - `/static/images/banks/scotiabank/favicon.ico`
- **PNG** - `/static/images/banks/scotiabank/favicon.png`

## Logo Display

### On the Landing Page
- Logos appear in a light grey section on the left side of the header
- If no logo is found, a placeholder with "Bank Logo - Place logo here" is shown
- Logos are automatically scaled to fit within 120x120px
- The section is responsive and stacks vertically on mobile devices

### Logo Validation
The system automatically:
- Checks if the specified logo file exists
- Falls back to alternative formats if the primary format is missing
- Shows a placeholder if no logo is found
- Validates logo paths and provides error messages

## Example: Adding Scotiabank Logo

1. **Create the directory:**
   ```bash
   mkdir -p src/web/static/images/banks/scotiabank/
   ```

2. **Add your logo files:**
   ```
   src/web/static/images/banks/scotiabank/
   ├── logo.svg    # Your Scotiabank logo
   └── favicon.svg # Your Scotiabank favicon
   ```

3. **That's it!** The system will automatically detect and display your logo.

## Troubleshooting

### Logo Not Showing
1. Check that the logo file exists in the correct path
2. Verify the file extension matches the path in the config
3. Ensure the file is readable by the web server
4. Check browser console for any 404 errors

### Logo Too Large/Small
- The system automatically scales logos to fit 120x120px
- For best results, provide logos that are at least 120x120px
- SVG format provides the best scaling quality

### Bank Not Detected
- Ensure the bank folder name exactly matches a preset configuration
- Check that the folder structure follows the pattern: `bank_statements/[bank_name]/[account_type]/[account_name]/`

## File Structure Summary

```
src/
├── config/
│   └── banks/
│       ├── scotiabank.json       # Preset Scotiabank configuration
│       └── README.md             # This file
└── web/
    └── static/
        └── images/
            └── banks/
                ├── default/      # Default logo and favicon
                └── scotiabank/   # User-added Scotiabank logos
```

## For Developers: Adding New Preset Banks

**Note:** This section is for developers who want to add new preset bank configurations.

### 1. Create Bank Configuration File

Copy `bank_template.json` and rename it to your bank name (e.g., `tdbank.json`):

```bash
cp bank_template.json tdbank.json
```

### 2. Update Configuration

Edit the configuration file and replace:
- `BANK_NAME` with your bank's internal name (e.g., `tdbank`)
- `Bank Display Name` with the display name (e.g., `TD Bank`)
- Colors and branding information
- PDF patterns specific to your bank's statement format

### 3. Create Logo Directory

Create the logo directory for the new bank:

```bash
mkdir -p ../../web/static/images/banks/tdbank/
```

### 4. Add Default Logos

Add default logo files to the new bank directory (these will be replaced by users with their own logos).

## Preset Bank Configurations

### Scotiabank
- **Folder name**: `scotiabank`
- **Display name**: Scotiabank
- **Colors**: Red (#E31837) and Navy Blue (#1B365D)
- **Logo path**: `/static/images/banks/scotiabank/logo.svg`
- **Supported account types**: Credit, Chequing, Savings 
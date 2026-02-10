# setup

This repo supports local setup for Windows (MSI) and macOS (DMG) without GitHub Actions.

## Windows MSI

Prereqs:
- Python 3.10+
- WiX v4+ (`dotnet tool install --global wix`)
- WiX extensions:
  - `wix extension add -g WixToolset.UI.wixext`
  - `wix extension add -g WixToolset.Util.wixext`

Build:
```powershell
setup\windows\build.ps1
```

Override version:
```powershell
setup\windows\build.ps1 -Version 1.2.3
```

Output:
- `dist\AirfoilFitter-x.y.z.msi`

## macOS DMG

Prereqs:
- Python 3.10+
- create-dmg (`brew install create-dmg`)

Build:
```bash
chmod +x setup/macos/build.sh
setup/macos/build.sh
```

Override version:
```bash
setup/macos/build.sh 1.2.3
```

Output:
- `dist/AirfoilFitter-x.y.z-arm64.dmg` or `dist/AirfoilFitter-x.y.z-x64.dmg`

## Versioning

Version is resolved in this order:
- explicit build argument
- latest git tag
- `1.0.0`


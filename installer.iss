[Setup]
AppName=COSMOS Galaxy Explorer
AppVersion=1.0
DefaultDirName={autopf}\CosmosGalaxyExplorer
DefaultGroupName=COSMOS Galaxy Explorer
OutputBaseFilename=CosmosGalaxy_Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "build_windows\cosmos.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "build_windows\glew32.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "build_windows\shaders\*"; DestDir: "{app}\shaders"; Flags: ignoreversion recursesubdirs
Source: "build_windows\data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs
Source: "build_windows\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs
Source: "build_windows\backend\*"; DestDir: "{app}\backend"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\COSMOS Galaxy Explorer"; Filename: "{app}\cosmos.exe"
Name: "{commondesktop}\COSMOS Galaxy Explorer"; Filename: "{app}\cosmos.exe"

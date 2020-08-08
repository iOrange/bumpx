@echo off
cl /EHsc /GS /GL /W3 /Gy /Zc:wchar_t /Qspectre /Gm- /O2 /sdl /Zc:inline /fp:precise /D "NDEBUG" /D "_CONSOLE" /D "_UNICODE" /D "UNICODE" /WX- /Zc:forScope /Gd /Oi /MD /std:c++17 ".\src\bumpx.cpp" /link /out:".\_build\bumpx.exe"

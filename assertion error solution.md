deal with: Debug Assertion Failed! Expression: __acrt_first_block == header 
when using opencv in MFC


���ֵ���detectMultiScaleʱ����heap���


�� Project -> "project" Properties -> Configuration Properties -> C/C++ -> code generation -> Runtime Library -> /MDd

Ȼ������ʾMFCû����shared DLL
��һ������

Project -> "project" Properties -> Configuration Properties -> C/C++ -> Advanced -> Show Includes:YES(/showIncludes)

Project -> "project" Properties -> Configuration Properties -> General -> Project Defaults -> Use of MFC :Use MFC in a shared DLL


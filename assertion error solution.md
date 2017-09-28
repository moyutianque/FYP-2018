deal with: Debug Assertion Failed! Expression: __acrt_first_block == header 
when using opencv in MFC


出现调用detectMultiScale时出现heap溢出


改 Project -> "project" Properties -> Configuration Properties -> C/C++ -> code generation -> Runtime Library -> /MDd

然而会提示MFC没有用shared DLL
下一步调：

Project -> "project" Properties -> Configuration Properties -> C/C++ -> Advanced -> Show Includes:YES(/showIncludes)

Project -> "project" Properties -> Configuration Properties -> General -> Project Defaults -> Use of MFC :Use MFC in a shared DLL


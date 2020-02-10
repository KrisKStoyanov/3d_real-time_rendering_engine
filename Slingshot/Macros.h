#pragma once
#define SAFE_SHUTDOWN(x) if(x != nullptr) { x->Shutdown(); delete x; x = nullptr; }
#define SAFE_RELEASE(x)  if(x!=nullptr) { x->Release(); x = nullptr; }
#define SAFE_DELETE_ARRAY(x) if(x!=nullptr) { delete [] x; x = nullptr; }
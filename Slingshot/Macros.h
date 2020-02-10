#pragma once
#define SAFE_SHUTDOWN(x) if(x != nullptr) { x->Shutdown(); delete x; x = nullptr; }
#define SAFE_RELEASE(x)  if(x!=nullptr) { x->Release(); x = nullptr; }
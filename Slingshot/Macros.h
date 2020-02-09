#pragma once
#define SAFE_SHUTDOWN(x) if(x != nullptr) { x->Shutdown(); delete x; x = nullptr; }

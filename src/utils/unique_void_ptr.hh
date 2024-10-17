
#ifndef UNIQUE_VOID_PTR_HH
#define UNIQUE_VOID_PTR_HH

#include <memory>

namespace utils{

// https://stackoverflow.com/questions/39288891/why-is-shared-ptrvoid-legal-while-unique-ptrvoid-is-ill-formed 
using unique_void_ptr = std::unique_ptr<void, void(*)(void const*)>;

template<typename T>
auto unique_void(T* ptr) -> unique_void_ptr
{
    return unique_void_ptr(ptr, [](void const* data) {
         T const* p = static_cast<T const*>(data);
         delete p;
    });
}

template<typename T, typename... Args>
auto make_unique_void(Args&&... args)
{
     return unique_void(new T(std::forward<Args>(args)...));
}


} // namespace utils
#endif
// ============================================================================================================
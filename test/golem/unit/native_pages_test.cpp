#include <sst/core/output.h>
#include <cassert>
#include <set>
#include <sst/elements/vanadis/os/vphysmemmanager.h>
int main(int, char**) {
    PhysMemManager manager(128ull*1024*1024);
    assert(manager.allocPage(PhysMemManager::FourKB)==0);
    assert(manager.reservePages(256,16));
    assert(!manager.reservePages(260,2));
    assert(!manager.reservePages(32760,16));
    assert(!manager.reservePages(1,0));
    std::set<unsigned> allocated{0};
    bool crossed=false;
    for(unsigned i=0;i<512;++i) {
        const unsigned page=manager.allocPage(PhysMemManager::FourKB);
        assert(page<256 || page>=272);
        assert(allocated.insert(page).second);
        crossed|=page>=272;
    }
    assert(crossed);
}

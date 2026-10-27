"""Execute production software tail ordering with synchronous completion stubs."""
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class TailScheduleTest(unittest.TestCase):
    def test_notification_order_and_failure(self):
        source = r'''
#include <cassert>
#include <vector>
#include <set>
#include "golem_tail_schedule.h"
int main() {
 for(unsigned logical=1;logical<=1024;++logical)
   assert(final_tail_producer(logical,1024)==4+(logical-1)/256+4*(((logical-1)%256)/64));
 assert(final_tail_producer(0,1024)==0);
 assert(final_tail_producer(1025,1024)==0);
 for(bool stream:{false,true})for(bool priority:{false,true})for(unsigned core:{4u,19u}) {
   std::vector<int> events;
   auto norm=[&]{events.push_back(1);return true;};
   auto publish=[&]{events.push_back(2);};
   auto wait=[&]{events.push_back(3);};
   auto lm=[&]{events.push_back(4);return true;};
   assert(run_final_tail(core,19,stream,priority,norm,publish,wait,lm));
   std::vector<int> expected;
   if(stream && priority && core!=19)expected.push_back(3);
   expected.push_back(1);
   if(stream && core==19)expected.push_back(2);
   expected.push_back(3);expected.push_back(4);
   assert(events==expected);
 }
 unsigned published=0,started=0;
 assert(!run_final_tail(19,19,true,false,[]{return false;},[&]{++published;},[]{},[&]{++started;return true;}));
 assert(!published && !started);
 // Only the token producer publishes once; other workers cannot authorize LM.
 for(unsigned core=4;core<20;++core)
   assert(run_final_tail(core,19,true,false,[]{return true;},[&]{++published;},[]{},[]{return true;}));
 assert(published==1);
 for(unsigned core=4;core<20;++core){
   std::set<uint64_t> destinations;bool initialized=false;
   publish_final_tail(core,0x200000,0x1028,4,
     [&](uint64_t epoch,uint64_t address){assert(epoch==4 && address==uint64_t(core)*0x200000+0x1028);initialized=true;},
     [&](uint64_t source,uint64_t target){assert(initialized && source==uint64_t(core)*0x200000+0x1028);
       assert(target!=source && target%0x200000==0x1028);assert(destinations.insert(target).second);});
   assert(destinations.size()==15);
 }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'test.cc').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(HERE),
                            str(p/'test.cc'), '-o', str(p/'test')], check=True)
            subprocess.run([str(p/'test')], check=True)


if __name__ == '__main__':
    unittest.main()

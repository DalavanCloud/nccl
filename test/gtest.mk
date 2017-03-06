### gtest. common def ##
BUILDDIR ?= $(abspath ../build)
BUILDRELDIRSELF = ../${BUILDDIR}
BUILDRELDIR = ../../${BUILDDIR}
GTEST_DIR ?= $(abspath googletest/googletest)
CPPFLAGS += -isystem $(GTEST_DIR)/include
CXXFLAGS += -pthread
NVCUFLAGS += -Xcompiler -pthread
GTEST_CPPFLAGS := -include gtest/gtest.h
GTEST_LIBS = $(BUILDRELDIR)/test/gtest.a $(BUILDRELDIR)/test/gtest_main.a
## gtest. build ##
GTEST_SRCS_ = $(addprefix $(GTEST_DIR)/,src/*.cc src/*.h include/gtest/*.h include/gtest/internal/*.h)
gtest.clean :
	rm -f $(addprefix $(BUILDRELDIR)/test/,gtest.a gtest_main.a gtest-all.o gtest_main.o)

$(BUILDRELDIRSELF)/test/gtest%.o : $(GTEST_SRCS_)
	@mkdir -p $(BUILDRELDIRSELF)/test
	$(CXX) -o $@ $(CPPFLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest$*.cc

$(BUILDRELDIRSELF)/test/gtest_main.a : $(BUILDRELDIRSELF)/test/gtest-all.o $(BUILDRELDIRSELF)/test/gtest_main.o
$(BUILDRELDIRSELF)/test/gtest.a : $(BUILDRELDIRSELF)/test/gtest-all.o
$(BUILDRELDIRSELF)/test/gtes%.a :
	$(AR) $(ARFLAGS) $@ $^

.PHONY: gtest
gtest : $(addprefix $(BUILDRELDIRSELF)/test/,gtest_main.a gtest.a)
###

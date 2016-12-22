### gtest. common def ##
BUILDDIR ?= $(abspath ../build)
GTEST_DIR ?= $(abspath googletest/googletest)
CPPFLAGS += -isystem $(GTEST_DIR)/include
CXXFLAGS += -pthread
NVCUFLAGS += -Xcompiler -pthread
GTEST_CPPFLAGS := -include gtest/gtest.h
GTEST_LIBS = $(BUILDDIR)/test/gtest.a $(BUILDDIR)/test/gtest_main.a
## gtest. build ##
GTEST_SRCS_ = $(addprefix $(GTEST_DIR)/,src/*.cc src/*.h include/gtest/*.h include/gtest/internal/*.h)
gtest.clean :
	rm -f $(addprefix $(BUILDDIR)/test/,gtest.a gtest_main.a gtest-all.o gtest_main.o)

$(BUILDDIR)/test/gtest%.o : $(GTEST_SRCS_)
	@mkdir -p $(BUILDDIR)/test
	$(CXX) -o $@ $(CPPFLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest$*.cc

$(BUILDDIR)/test/gtest_main.a : $(BUILDDIR)/test/gtest-all.o $(BUILDDIR)/test/gtest_main.o
$(BUILDDIR)/test/gtest.a : $(BUILDDIR)/test/gtest-all.o
$(BUILDDIR)/test/gtes%.a :
	$(AR) $(ARFLAGS) $@ $^

.PHONY: gtest
gtest : $(addprefix $(BUILDDIR)/test/,gtest_main.a gtest.a)
###

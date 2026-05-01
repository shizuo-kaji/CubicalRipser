# Configurable settings (override on command line, e.g. make MODE=debug)
MODE        ?= release
CXX         ?= clang++
CXXSTD      ?= c++20

# Architectures: set to "arm64", "x86_64", or "arm64 x86_64" for universal
ARCHS       ?= $(shell uname -m)

# Common warning & language flags
WARNFLAGS   = -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow

# Optimization / debug
ifeq ($(MODE),release)
	OPTFLAGS  = -O3 -DNDEBUG
else ifeq ($(MODE),relassert)
	OPTFLAGS  = -O3
else
	OPTFLAGS  = -O0 -g -fno-omit-frame-pointer
endif

# Architecture flags
ARCHFLAGS   = $(addprefix -arch ,$(ARCHS))

# Base compile flags
CXXFLAGS    = $(OPTFLAGS) $(WARNFLAGS) -std=$(CXXSTD) $(ARCHFLAGS)

# Enable automatic dependency generation
DEPFLAGS    = -MMD -MP

# Linker flags (extend if needed, e.g. -L/path -lfoo)
LDFLAGS     = $(ARCHFLAGS)

TARGET1     = cubicalripser
TARGET2     = tcubicalripser

SRCS_COMMON = coboundary_enumerator.cpp joint_pairs.cpp compute_pairs.cpp ph_2d.cpp
SRCS1       = cubicalripser.cpp dense_cubical_grids.cpp $(SRCS_COMMON)
SRCS2       = cubicalripser.cpp dense_cubical_grids_T.cpp $(SRCS_COMMON)

OBJDIR      = build
OBJS1       = $(SRCS1:%=$(OBJDIR)/%.o)
OBJS2       = $(SRCS2:%=$(OBJDIR)/%.o)
DEPS        = $(OBJS1:.o=.d) $(OBJS2:.o=.d)

.DEFAULT_GOAL := all

.PHONY: all clean dirs
all: $(TARGET1) $(TARGET2)

dirs:
	@mkdir -p $(OBJDIR)

$(TARGET1): $(OBJS1)
	$(CXX) $(LDFLAGS) -o $@ $^

$(TARGET2): $(OBJS2)
	$(CXX) $(LDFLAGS) -o $@ $^

# Pattern rule for object files
$(OBJDIR)/%.cpp.o: %.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# Shorter suffix-style alternative (maps %.cpp to build/%.cpp.o)
%.o: ;

-include $(DEPS)

clean:
	rm -rf $(OBJDIR) $(TARGET1) $(TARGET2)

BUILDDIR ?= $(abspath ../build)
DOCDIR := $(BUILDDIR)/doc

FILES := index.html css/doc.css img/Reduce.png img/AllReduce.png img/ReduceScatter.png img/Bcast.png img/AllGather.png
DOCFILES := $(FILES:%=$(DOCDIR)/%)

build: $(DOCFILES)

$(DOCDIR)/index.html : index0.html
	mkdir -p $(DOCDIR)
	./index_titles.py $< > $@

$(DOCDIR)/css/%.css : css/%.css
	mkdir -p $(DOCDIR)/css
	cp $< $@

$(DOCDIR)/img/%.png : img/%.png
	mkdir -p $(DOCDIR)/img
	cp $< $@

clean :
	rm -Rf $(DOCDIR)

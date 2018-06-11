GETTEXT_DOMAIN=human-icon-theme

build: index.theme

# intltool merging
index.theme:
	for f in `find -name *.icon.in -o -name *.theme.in`; do \
	    intltool-merge -d -u po $$f $${f%.in}; \
	done
	cd po; intltool-update -p --verbose -g $(GETTEXT_DOMAIN)

clean:
	find -name *.icon -exec rm '{}' \;
	rm -f po/*.pot index.theme

install: index.theme
	find [0-9]* scalable \( -name '*.png' -o -name '*.svg' -o -name '*.icon' \) -exec \
	    install -D -m 644 '{}' $$DESTDIR/usr/share/icons/Human/'{}' \;

	# run icon-naming-utils
	for d in $$DESTDIR/usr/share/icons/Human/*; do \
	    (cd $$d; for c in *; do /usr/lib/icon-naming-utils/icon-name-mapping -c $$c; done); \
	done

	# install mo files
	for f in po/*.po; do \
	    D=$$DESTDIR/usr/share/locale/`basename $$f .po`/LC_MESSAGES/; \
	    install -d -m 755 $$D; \
	    msgfmt -o $$D/$(GETTEXT_DOMAIN).mo $$f; \
	done

	# install index
	install -D -m 644 index.theme $$DESTDIR/usr/share/icons/Human/index.theme

.PHONY: build i18n clean install

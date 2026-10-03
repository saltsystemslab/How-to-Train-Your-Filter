/*
Registry of filters available to the generic experiment harnesses.
To add a filter, write its adapter (see src/aqf_adapter.c), declare its ops table in
include/filter_iface.h, and add it to the list below.
*/
#include <string.h>

#include "include/filter_iface.h"

static const filter_ops *registered_filters[] = {
	&aqf_ops,
	&stacked_ops,
};

const filter_ops *find_filter(const char *name) {
	for (size_t i = 0; i < sizeof(registered_filters) / sizeof(registered_filters[0]); i++) {
		if (strcmp(registered_filters[i]->name, name) == 0) {
			return registered_filters[i];
		}
	}
	return NULL;
}

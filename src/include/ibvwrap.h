
// Dynamically handle dependencies on IB verbs

/* Extracted from infiniband/verbs.h*/

enum ibv_node_type {
	IBV_NODE_UNKNOWN	= -1,
	IBV_NODE_CA 		= 1,
	IBV_NODE_SWITCH,
	IBV_NODE_ROUTER,
	IBV_NODE_RNIC
};

enum ibv_transport_type {
	IBV_TRANSPORT_UNKNOWN	= -1,
	IBV_TRANSPORT_IB	= 0,
	IBV_TRANSPORT_IWARP
};

enum {
  IBV_SYSFS_NAME_MAX	= 64,
  IBV_SYSFS_PATH_MAX	= 256
};

struct ibv_device;
struct ibv_context;

struct ibv_device_ops {
	struct ibv_context *	(*alloc_context)(struct ibv_device *device, int cmd_fd);
	void			(*free_context)(struct ibv_context *context);
};

struct ibv_device {
  struct ibv_device_ops	ops;
  enum ibv_node_type	node_type;
  enum ibv_transport_type	transport_type;
  /* Name of underlying kernel IB device, eg "mthca0" */
  char			name[IBV_SYSFS_NAME_MAX];
  /* Name of uverbs device, eg "uverbs0" */
  char			dev_name[IBV_SYSFS_NAME_MAX];
  /* Path to infiniband_verbs class device in sysfs */
  char			dev_path[IBV_SYSFS_PATH_MAX];
  /* Path to infiniband class device in sysfs */
  char			ibdev_path[IBV_SYSFS_PATH_MAX];
};

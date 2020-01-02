#!/bin/bash
while (( "$#" )); do
  case "$1" in
    -s|--subject)
      SUBJECT=$2
      shift 2
      ;;
    -n|--session)
      SESSION=$2
      shift 2
      ;;
    -r|--run)
      RUN=$2
      shift 2
      ;;
    -t|--task)
      TASK=$2
      shift 2
      ;;
    -b|--behf)
      BEHF=$2
      shift 2
      ;;
    -e|--eegf)
      EEGF=$2
      shift 2
      ;;
    -d|--dir)
      DIR=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done

python save2bids.py $SUBJECT $SESSION $RUN $TASK $EEGF $DIR
my_dir=`pwd`

if [ ! -z "$TASK" ]
then
	echo "Processing behavior"
	$MATLAB_ROOT/bin/matlab -nodisplay -nosplash -nodesktop -r "cd '$my_dir'; slow_fast_mat2csv('$BEHF'); quit;"
	mkdir -p "$DIR/sub-$SUBJECT/ses-$SESSION/beh"
	cp ${BEHF%.mat}.tsv $DIR/sub-$SUBJECT/ses-$SESSION/beh/sub-${SUBJECT}_ses-${SESSION}_task-${TASK}_run-${RUN}_beh.tsv
	python add_subject_info.py $SUBJECT $DIR ${BEHF%.mat}parameters.tsv
fi